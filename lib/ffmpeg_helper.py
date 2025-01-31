import os
import time
import json
import glob
import subprocess
import shlex
from pathlib import Path
from urllib.parse import urlparse
from lib import util

def probe_stream(video_url):
    """
    비디오 스트림 정보를 추출하는 함수
    
    Args:
        video_url: 비디오 파일 경로 또는 URL
    Returns:
        stream_info: 비디오 스트림 정보가 담긴 딕셔너리
    """
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    stream_info_file = os.path.join(video_dir, 'stream_info.json')

    # URL 스키마 검증
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # 로컬 파일 존재 여부 확인
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')
        
    util.mkdir(video_dir)
    
    # 이미 생성된 stream_info.json 파일이 있는지 확인
    if os.path.exists(stream_info_file):
        with open(stream_info_file, 'r', encoding="utf-8") as f:
            stream_info = json.loads(f.read())
        print(f"  probe_stream: found stream_info.json. SKIPPING...")
        return stream_info

    # ffprobe로 스트림 정보 추출
    command_string = f'ffprobe -v quiet -print_format json -show_format -show_streams {shlex.quote(video_url)}'
    child_process = subprocess.Popen(
        shlex.split(command_string),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = child_process.communicate()

    stream_info = json.loads(str(stdout, 'utf-8'))
    stream_info['format']['filename'] = video_file

    # 비디오 스트림 정보 파싱
    video_stream = list(filter(lambda x: (x['codec_type'] == 'video'), stream_info['streams']))[0]

    # 비디오 속성 추출
    progressive = bool('field_order' not in video_stream or video_stream['field_order'] == 'progressive')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration_ms = int(float(video_stream['duration']) * 1000)
    num_frames = int(video_stream['nb_frames'])
    framerate = util.to_fraction(video_stream['r_frame_rate'])
    sample_aspect_ratio = util.to_fraction(video_stream['sample_aspect_ratio'])
    display_aspect_ratio = util.to_fraction(video_stream['display_aspect_ratio'])
    display_width = int((width * sample_aspect_ratio.numerator) / sample_aspect_ratio.denominator)

    # 비디오 스트림 정보 저장
    stream_info['video_stream'] = {
        'duration_ms': duration_ms,
        'num_frames': num_frames,
        'framerate': (framerate.numerator, framerate.denominator),
        'progressive': progressive,
        'sample_aspect_ratio': (sample_aspect_ratio.numerator, sample_aspect_ratio.denominator),
        'display_aspect_ratio': (display_aspect_ratio.numerator, display_aspect_ratio.denominator),
        'encoded_resolution': (width, height),
        'display_resolution': (display_width, height),
    }

    util.save_to_file(stream_info_file, stream_info)
    return stream_info

def extract_frames(video_url, stream_info, max_res = (750, 500)):
    """
    비디오에서 프레임을 추출하는 함수
    
    Args:
        video_url: 비디오 파일 경로 또는 URL
        stream_info: 비디오 스트림 정보
        max_res: 최대 해상도 (width, height)
    Returns:
        jpeg_frames: 추출된 JPEG 프레임 파일 목록
    """
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    
    # URL 스키마 검증
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # 로컬 파일 존재 여부 확인
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')

    frame_dir = os.path.join(video_dir, 'frames')
    
    # 이미 추출된 프레임이 있는지 확인
    if os.path.exists(frame_dir):
        jpeg_frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
        print(f"  extract_frames: found {len(jpeg_frames)} frames. SKIPPING...")
        return jpeg_frames

    util.mkdir(frame_dir)

    t0 = time.time()
    video_filters = []
    video_stream = stream_info['video_stream']

    # 디인터레이싱 필요 여부 확인
    progressive = video_stream['progressive']
    if not progressive:
        video_filters.append('yadif')

    # 이미지 다운스케일링
    dw, dh = video_stream['display_resolution']
    factor = max((max_res[0] / dw), (max_res[1] / dh))
    w = round((dw * factor) / 2) * 2
    h = round((dh * factor) / 2) * 2
    video_filters.append(f"scale={w}x{h}")

    # ffmpeg 명령어 구성
    command = [
        'ffmpeg',
        '-v',
        'quiet',
        '-i',
        shlex.quote(video_url),
        '-vf',
        f"{','.join(video_filters)}",
        '-r',
        str(1),
        f"{shlex.quote(frame_dir)}/frames.%07d.jpg"
    ]

    print(f"  Resizing: {dw}x{dh} -> {w}x{h} (Progressive? {progressive})")
    print(f"  Command: {command}")
    
    # ffmpeg 실행
    subprocess.run(
        command,
        shell=False,
        stdout=subprocess.DEVNULL
    )

    t1 = time.time()
    print(f"  extract_frames: elapsed {round(t1 - t0, 2)}s")

    # 추출된 JPEG 파일 목록 반환
    jpeg_frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
    return jpeg_frames

def extract_audio(video_url):
    """
    비디오에서 오디오를 추출하는 함수
    
    Args:
        video_url: 비디오 파일 경로 또는 URL
    Returns:
        wav_file: 추출된 WAV 파일 경로
    """
    t0 = time.time()
    
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    
    # URL 스키마 검증
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # 로컬 파일 존재 여부 확인
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')

    audio_dir = Path(urlparse(video_url).path).stem
    wav_file = os.path.join(audio_dir, 'audio.wav')

    # 이미 추출된 WAV 파일이 있는지 확인
    if os.path.exists(wav_file):
        print(f"  extract_audio: found audio.wav. SKIPPING...")
        return wav_file

    # ffmpeg 오디오 추출 설정
    bitrate = '96k'
    sampling_rate = 16000
    channel = 1
    
    # ffmpeg 명령어 구성
    command = [
        'ffmpeg',
        '-i',
         shlex.quote(video_url),
        '-vn',
        '-c:a',
        'pcm_s16le',
        '-ab',
        bitrate,
        '-ar',
        str(sampling_rate),
        '-ac',
        str(channel),
        wav_file
    ]
    print(command)
    
    # ffmpeg 실행
    subprocess.run(
        command,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    t1 = time.time()
    print(f"  extract_audio: elapsed {round(t1 - t0, 2)}s")
    return wav_file

def create_lowres_video(video_url, stream_info, max_res = (360, 202)):
    """
    저해상도 비디오를 생성하는 함수
    
    Args:
        video_url: 비디오 파일 경로 또는 URL
        stream_info: 비디오 스트림 정보
        max_res: 최대 해상도 (width, height)
    Returns:
        low_res_video_file: 생성된 저해상도 비디오 파일 경로
    """
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    
    # URL 스키마 검증
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # 로컬 파일 존재 여부 확인
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')
    
    low_res_video_file = os.path.join(video_dir, 'lowres_video.mp4')

    # 이미 생성된 저해상도 비디오가 있는지 확인
    if os.path.exists(low_res_video_file):
        print(f"  create_lowres_video: found lowres_video.mp4. SKIPPING...")
        return low_res_video_file

    util.mkdir(video_dir)

    video_stream = stream_info['video_stream']
    video_filters = []
    
    # 디인터레이싱 필요 여부 확인
    progressive = video_stream['progressive']
    if not progressive:
        video_filters.append('yadif')

    # 이미지 다운스케일링
    dw, dh = video_stream['display_resolution']
    factor = max((max_res[0] / dw), (max_res[1] / dh))
    w = round((dw * factor) / 2) * 2
    h = round((dh * factor) / 2) * 2
    video_filters.append(f"scale={w}x{h}")

    # ffmpeg 명령어 구성
    command = [
        'ffmpeg',
        '-v',
        'quiet',
        '-i',
        shlex.quote(video_url),
        '-vf',
        f"{','.join(video_filters)}",
        '-ac',
        str(2),
        '-ab',
        '64k',
        '-ar',
        str(44100),
        f"{low_res_video_file}"
    ]

    print(f"  Downscaling: {dw}x{dh} -> {w}x{h} (Progressive? {progressive})")
    print(f"  Command: {command}")

    t0 = time.time()
    
    # ffmpeg 실행
    subprocess.run(
        command,
        shell=False,
        stdout=subprocess.DEVNULL
    )

    t1 = time.time()
    print(f"  downscale_video: elapsed {round(t1 - t0, 2)}s")

    return low_res_video_file