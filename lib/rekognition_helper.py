import librosa
import boto3
import os
import time
import math 
import numpy as np
import pandas as pd
import cv2
from termcolor import colored

# AWS Rekognition 클라이언트 생성
rekognition_client = boto3.client('rekognition')

def round_down_to_nearest_half_second(timestamp_millis):
    # 밀리초를 초 단위로 변환하고 0.5초 단위로 내림
    rounded_seconds = math.floor(timestamp_millis / 1000.0 * 2) / 2.0
    return float(rounded_seconds)

def populate_segment_indicator_dict(results):
    # 비디오 세그먼트 정보를 저장할 딕셔너리 생성
    segment_indicator_dict = {}
    
    # 비디오 전체 길이(밀리초) 가져오기
    duration=results['VideoMetadata'][0]['DurationMillis']
    duration=round_down_to_nearest_half_second(duration)
    segments=results['Segments'] 
    
    # 0.5초 간격으로 딕셔너리 초기화
    current=0
    while current < duration:
        segment_indicator_dict[current]=0 
        current=current+0.5
    
    # 각 세그먼트의 시작 시간을 딕셔너리에 저장
    for segment in segments:
        start_time=segment['StartTimestampMillis']
        start_time_segment = round_down_to_nearest_half_second(segment['StartTimestampMillis'])
        segment_indicator_dict[start_time_segment]=start_time
    return segment_indicator_dict

def detect_video_segments(bucket_name, key, video_file, interval=0.5):
    # AWS Rekognition을 사용하여 비디오 세그먼트 감지 시작
    response = rekognition_client.start_segment_detection(
        Video={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': os.path.join(key, video_file)
            }
        },
        Filters={
             'ShotFilter': {
            'MinSegmentConfidence': 70
        }
        },
        SegmentTypes=['SHOT'] 
    )

    job_id = response['JobId']
    return job_id

def get_results(job_id):
    # 세그먼트 감지가 완료될 때까지 대기
    while True:
        result = rekognition_client.get_segment_detection(JobId=job_id)
        if result['JobStatus'] in ['SUCCEEDED', 'FAILED']:
            break
        time.sleep(5)  # 5초 간격으로 상태 체크

    # 세그먼트 결과 분석
    segments = result
    segment_indicator_dict = populate_segment_indicator_dict(segments)
    return segment_indicator_dict

def extract_volume_levels(audio_file_path):
    # librosa를 사용하여 오디오 파일 로드
    audio, sr = librosa.load(audio_file_path, sr=None)

    # 500밀리초 간격으로 볼륨 레벨 계산
    frame_size = int(sr / 2)  # 500밀리초
    num_frames = len(audio) // frame_size

    volume_data = {"Time": [], "Volume": []}

    # 각 프레임별 볼륨 레벨 계산
    for i in range(num_frames):
        frame = audio[i * frame_size: (i + 1) * frame_size]
        time_in_seconds = i * 0.5  # 500밀리초는 0.5초

        # 볼륨 레벨 정규화 (0-255 범위)
        volume_level = np.mean(np.abs(frame))
        normalized_volume = int((volume_level / np.max(audio)) * 255)
        
        volume_data["Time"].append(time_in_seconds)
        volume_data["Volume"].append(normalized_volume)

    # DataFrame 생성
    df = pd.DataFrame(volume_data)
    return df 

def extract_frame(video_path, frame_milliseconds, output_path):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 원하는 밀리초 위치로 프레임 설정
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_milliseconds)

    # 지정된 위치에서 프레임 읽기
    success, frame = cap.read()

    # 프레임 추출 성공 시 저장
    if success:
        cv2.imwrite(output_path, frame)
        print(f"Frame at {frame_milliseconds} milliseconds extracted and saved to {output_path}")
    else:
        print(f"Failed to extract frame at {frame_milliseconds} milliseconds")

    # 비디오 캡처 객체 해제
    cap.release()


def estimate_rekognition_cost(duration_ms):
    """
    Rekognition Shot Detection 비용을 추정하는 함수
    
    Args:
        duration_ms: 비디오 길이(밀리초)
    Returns:
        dict: 비용 추정 정보
    """
    rekognition_per_min = 0.05
    duration_minutes = math.ceil(duration_ms / 60000)
    estimated_cost = round(rekognition_per_min * duration_minutes, 4)

    return {
        'cost_per_min': rekognition_per_min,
        'duration': round(duration_ms / 1000, 2),
        'estimated_cost': estimated_cost,
    }


def display_rekognition_cost(duration_ms):
    """
    Rekognition 비용을 출력하는 함수
    
    Args:
        duration_ms: 비디오 길이(밀리초)
    Returns:
        rekognition_cost: 비용 추정 정보
    """
    rekognition_cost = estimate_rekognition_cost(duration_ms)
    print('\nEstimated cost for Rekognition Shot Detection:', 
          colored(f"${rekognition_cost['estimated_cost']}", 'green'), 
          f"in us-west-2 region with duration: {rekognition_cost['duration']}s")
    return rekognition_cost