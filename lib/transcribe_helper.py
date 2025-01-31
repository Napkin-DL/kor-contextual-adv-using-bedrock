
import os
import boto3
import time
from pathlib import Path
import requests
import re
import pandas as pd
from termcolor import colored

def url_retrieve(url: str, outfile: Path):
    """
    URL에서 파일을 다운로드하여 저장하는 함수
    
    Args:
        url: 다운로드할 파일의 URL
        outfile: 저장할 파일 경로
    """
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    with open(outfile,'wb') as f:
        f.write(r.content)

def transcribe(bucket, path, file, media_format="mp4", language_code="en-US", verbose=True):
    """
    AWS Transcribe를 사용하여 음성을 텍스트로 변환하는 함수
    
    Args:
        bucket: S3 버킷 이름
        path: S3 경로
        file: 변환할 파일명
        media_format: 미디어 포맷 (기본값: mp4)
        language_code: 언어 코드 (기본값: en-US)
        verbose: 상세 출력 여부
    Returns:
        transcribe_response: 변환 작업 응답 객체
    """
    # 이미 transcript가 존재하는지 확인
    video_dir = Path(file).stem
    if os.path.exists(os.path.join(video_dir, 'transcript.vtt')):
        print(colored(f"Transcript already exists for {file}", 'yellow'))
        return None

    # 변환 작업 시작
    transcribe_response = start_transcription_job(
        bucket, 
        path,
        file, 
        media_format, 
        language_code
    )

    # 작업 완료 대기
    transcribe_response = wait_for_transcription_job(
        transcribe_response['TranscriptionJob']['TranscriptionJobName'], 
        verbose
    )

    return transcribe_response

def start_transcription_job(bucket, path, file, media_format="mp4", language_code="en-US"):
    """
    AWS Transcribe 작업을 시작하는 함수
    
    Args:
        bucket: S3 버킷 이름
        path: S3 경로 
        file: 변환할 파일명
        media_format: 미디어 포맷
        language_code: 언어 코드
    Returns:
        response: AWS Transcribe 응답
    """
    # 랜덤 작업명 생성
    job_name = '-'.join([
        Path(file).stem,
        os.urandom(4).hex(),
    ])

    key = path+'/'+file
    transcribe_client = boto3.client('transcribe')

    # Transcribe 작업 시작
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode=language_code,
        MediaFormat=media_format,
        Media={
            'MediaFileUri': f"s3://{bucket}/{key}",
        },
        Subtitles={
            'Formats': ['vtt'],
            'OutputStartIndex': 1
        },
    )

    return response

def wait_for_transcription_job(job_name, verbose=True):
    """
    Transcribe 작업 완료를 대기하는 함수
    
    Args:
        job_name: 작업명
        verbose: 상세 출력 여부
    Returns:
        response: 작업 상태 응답
    """
    transcribe_client = boto3.client('transcribe')

    while True:
        try:
            response = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            transcription_job_status = response['TranscriptionJob']['TranscriptionJobStatus']
            if verbose: 
                print(f"wait_for_transcription_job: status = {transcription_job_status}")
            if transcription_job_status in ['COMPLETED', 'FAILED']:
                return response
            time.sleep(4)
        except Exception as e:
            print(f"Error fetching transcription job status: {e}")
            raise

def estimate_transcribe_cost(duration_ms):
    """
    Transcribe 비용을 추정하는 함수
    
    Args:
        duration_ms: 오디오 길이(밀리초)
    Returns:
        dict: 비용 추정 정보
    """
    transcribe_batch_per_min = 0.02400
    transcribe_cost = round(transcribe_batch_per_min * (duration_ms / 60000), 4)

    return {
        'cost_per_min': transcribe_batch_per_min,
        'duration': round(duration_ms / 1000, 2),
        'estimated_cost': transcribe_cost,
    }

def display_transcription_cost(duration_ms):
    """
    Transcribe 비용을 출력하는 함수
    
    Args:
        duration_ms: 오디오 길이(밀리초)
    Returns:
        transcribe_cost: 비용 추정 정보
    """
    transcribe_cost = estimate_transcribe_cost(duration_ms)
    print('\nEstimated cost to Transcribe video:', 
          colored(f"${transcribe_cost['estimated_cost']}", 'green'), 
          f"in us-west-2 region with duration: {transcribe_cost['duration']}s")
    return transcribe_cost

def download_vtt(response, output_dir = ''):
    """
    VTT 자막 파일을 다운로드하는 함수
    
    Args:
        response: Transcribe 작업 응답
        output_dir: 출력 디렉토리
    Returns:
        output_file: 저장된 파일 경로
    """
    output_file = os.path.join(output_dir, 'transcript.vtt')
    if os.path.exists(output_file):
        return output_file

    subtitle_uri = response['TranscriptionJob']['Subtitles']['SubtitleFileUris'][0]
    url_retrieve(subtitle_uri, output_file)
    return output_file

def download_transcript(response, output_dir = ''):
    """
    Transcript JSON 파일을 다운로드하는 함수
    
    Args:
        response: Transcribe 작업 응답
        output_dir: 출력 디렉토리
    Returns:
        output_file: 저장된 파일 경로
    """
    output_file = os.path.join(output_dir, 'transcript.json')
    if os.path.exists(output_file):
        return output_file

    transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
    url_retrieve(transcript_uri, output_file)
    return output_file

def parse_file(file_path):
    """
    VTT 파일을 파싱하여 DataFrame으로 변환하는 함수
    
    Args:
        file_path: VTT 파일 경로
    Returns:
        df: 파싱된 데이터가 담긴 DataFrame
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # 타임스탬프와 텍스트를 추출하기 위한 정규식 패턴
    pattern = re.compile(r'(\d+:\d+:\d+.\d+) --> (\d+:\d+:\d+.\d+)\n(.+?)\n\n', re.DOTALL)
    matches = re.findall(pattern, content)

    # 0.5초 간격으로 텍스트 출현 여부를 저장할 딕셔너리
    time_dict = {}

    # 매치된 결과 처리
    for match in matches:
        start_time, end_time, _ = match
        start_seconds = convert_to_seconds(start_time)
        end_seconds = convert_to_seconds(end_time)

        current_time = start_seconds
        while current_time < end_seconds:
            time_key = round(current_time * 2) / 2
            time_dict.setdefault(time_key, True)
            current_time += 0.5

    # 전체 구간에 대해 텍스트가 없는 구간은 False로 설정
    total_duration = convert_to_seconds(matches[-1][1])
    for time_key in range(0, int(total_duration * 2) + 1):
        time_key /= 2
        if time_key not in time_dict:
            time_dict[time_key] = False

    # DataFrame 생성 및 반환
    sorted_time_dict = dict(sorted(time_dict.items()))
    df = pd.DataFrame(list(sorted_time_dict.items()), columns=['Time', 'Speech Appears'])
    df['Speech Appears'] = df['Speech Appears'].map(lambda x: '1' if x else '0')
    return df 

def convert_to_seconds(time_str):
    """
    시간 문자열을 초 단위로 변환하는 함수
    
    Args:
        time_str: 시간 문자열 (HH:MM:SS.mmm 형식)
    Returns:
        float: 초 단위 시간
    """
    h, m, s, ms = map(int, time_str.replace('.', ':').split(':'))
    return h * 3600 + m * 60 + s + ms / 1000