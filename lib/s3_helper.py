import os
import boto3
import sagemaker

def upload_object(bucket, prefix, file):
    """
    파일을 S3 버킷에 업로드하는 함수
    
    Args:
        bucket: S3 버킷 이름
        prefix: S3 경로 접두사 
        file: 업로드할 로컬 파일 경로
    Returns:
        dict: S3 업로드 응답 객체
    
    Example:
        >>> upload_object('my-bucket', 'data/videos', 'sample.mp4')
    """
    # S3 키(경로) 생성
    key = os.path.join(prefix, file)

    # S3 클라이언트 생성
    s3_client = boto3.client('s3')

    # 파일 업로드
    with open(file, "rb") as f:
        response = s3_client.put_object(
            Body=f,
            Bucket=bucket,
            Key=key,
        )
        
    return response