import json
import os
import boto3
import faiss
from functools import cmp_to_key
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image
from pathlib import Path
from termcolor import colored
from lib import frames
from lib import util

# Titan 모델 ID와 가격 설정
TITAN_MODEL_ID = 'amazon.titan-embed-image-v1'
TITAN_PRICING = 0.00006

def batch_generate_embeddings(jpeg_files, output_dir = ''):
    """
    이미지 파일들의 임베딩을 생성하는 함수
    
    Args:
        jpeg_files: JPEG 파일 목록
        output_dir: 출력 디렉토리
    Returns:
        list: 프레임 임베딩 정보
    """
    output_file = os.path.join(output_dir, 'frame_embeddings.json')
    
    # 기존 임베딩 파일이 있는지 확인
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:        
            frame_embeddings = json.load(f)
            return frame_embeddings

    frame_embeddings = []
    
    # Bedrock 클라이언트 설정
    titan_model_id = TITAN_MODEL_ID
    accept = 'application/json'
    content_type = 'application/json'
    bedrock_runtime_client = boto3.client(service_name='bedrock-runtime')

    # 각 이미지 파일에 대해 임베딩 생성
    for jpeg_file in jpeg_files:
        with Image.open(jpeg_file) as image:
            input_image = frames.image_to_base64(image)

        # 모델 파라미터 설정
        model_params = {
            'inputImage': input_image,
            'embeddingConfig': {
                'outputEmbeddingLength': 384
            }
        }

        # 모델 호출
        body = json.dumps(model_params)
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=titan_model_id,
            accept=accept,
            contentType=content_type
        )
        response_body = json.loads(response.get('body').read())

        # 프레임 번호 추출 및 임베딩 저장
        frame_no = int(Path(jpeg_file).stem.split('.')[1]) - 1
        frame_embeddings.append({
            'file': jpeg_file,
            'frame_no': frame_no,
            'embedding': response_body['embedding']
        })

    util.save_to_file(output_file, frame_embeddings)
    return frame_embeddings

def display_embedding_cost(frame_embeddings):
    """
    임베딩 생성 비용을 계산하고 표시하는 함수
    
    Args:
        frame_embeddings: 프레임 임베딩 정보
    Returns:
        dict: 비용 정보
    """
    per_image_embedding = TITAN_PRICING
    estimated_cost = per_image_embedding * len(frame_embeddings)

    print('\n')
    print('========================================================================')
    print('Estimated cost:', colored(f"${round(estimated_cost, 4)}", 'green'), 
          f"in us-west-2 region with {len(frame_embeddings)} embeddings")
    print('========================================================================')

    return {
        'per_image_embedding': per_image_embedding,
        'estimated_cost': estimated_cost,
        'num_embeddings': len(frame_embeddings)
    }

def create_index(dimension):
    """
    FAISS 인덱스를 생성하는 함수
    
    Args:
        dimension: 임베딩 차원
    Returns:
        faiss.Index: 생성된 인덱스
    """
    index = faiss.IndexFlatIP(dimension) # cosine similarity
    return index

def index_frames(index, frame_embeddings):
    """
    프레임 임베딩을 인덱스에 추가하는 함수
    
    Args:
        index: FAISS 인덱스
        frame_embeddings: 프레임 임베딩 정보
    Returns:
        faiss.Index: 업데이트된 인덱스
    """
    for item in frame_embeddings:
        embedding = np.array([item['embedding']])
        index.add(embedding)
    return index

def cosine_similarity(a, b):
    """
    코사인 유사도를 계산하는 함수
    
    Args:
        a, b: 비교할 벡터
    Returns:
        float: 코사인 유사도
    """
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def cmp_min_max(a, b):
    """
    시작 shot_id와 종료 shot_id를 기준으로 정렬하는 비교 함수
    
    Args:
        a, b: 비교할 항목
    Returns:
        int: 비교 결과
    """
    if a[0] < b[0]:
        return -1
    if a[0] > b[0]:
        return 1
    return b[1] - a[1]

def search_similarity(index, frame, k = 20, min_similarity = 0.80, time_range = 30):
    """
    유사한 프레임을 검색하는 함수
    
    Args:
        index: FAISS 인덱스
        frame: 검색할 프레임 정보
        k: 검색할 최대 결과 수
        min_similarity: 최소 유사도 임계값
        time_range: 시간 범위 제한
    Returns:
        list: 유사한 프레임 목록
    """
    idx = int(frame['frame_no'])
    embedding = np.array([frame['embedding']])

    # k-최근접 이웃 검색
    D, I = index.search(embedding, k)

    # 유사도 정보 구성
    similar_frames = [
        {
            'idx': int(i),
            'similarity': float(d)
        } for i, d in zip(I[0], D[0])
    ]

    # 최소 유사도 필터링
    similar_frames = list(
        filter(
            lambda x: x['similarity'] > min_similarity,
            similar_frames
        )
    )

    # 인덱스 기준 정렬
    similar_frames = sorted(similar_frames, key=lambda x: x['idx'])

    # 시간 범위 기준 필터링
    filtered_by_time_range = [{
        'idx': idx,
        'similarity': 1.0
    }]

    for i in range(0, len(similar_frames)):
        prev = filtered_by_time_range[-1]
        cur = similar_frames[i]

        if abs(prev['idx'] - cur['idx']) < time_range:
            filtered_by_time_range.append(cur)

    return filtered_by_time_range