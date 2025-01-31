import os
import base64
import copy
from io import BytesIO
from functools import cmp_to_key
from PIL import Image, ImageDraw
from IPython.display import display
from lib import util
from lib import embeddings

def image_to_base64(image):
    """
    이미지를 base64 문자열로 변환하는 함수
    
    Args:
        image: PIL Image 객체
    Returns:
        str: base64 인코딩된 이미지 문자열
    """
    buff = BytesIO()
    image.save(buff, format='JPEG')
    return base64.b64encode(buff.getvalue()).decode('utf8')

def skip_frames(frames, max_frames = 80):
    """
    프레임 수를 제한하기 위해 일부 프레임을 건너뛰는 함수
    
    Args:
        frames: 프레임 목록
        max_frames: 최대 프레임 수
    Returns:
        list: 선택된 프레임 목록
    """
    if len(frames) < max_frames:
        return frames

    skip_step = max(round(len(frames) / max_frames), 2)
    return frames[::skip_step]

def create_grid_image(image_files, max_ncol = 10, border_width = 2):
    """
    이미지들을 그리드 형태로 배치하는 함수
    
    Args:
        image_files: 이미지 파일 목록
        max_ncol: 최대 열 수
        border_width: 테두리 두께
    Returns:
        Image: 그리드 이미지
    """
    should_resize = len(image_files) > 50

    # 기준 이미지 크기 가져오기
    with Image.open(image_files[0]) as image:
        width, height = image.size

    # 그리드 크기 계산
    ncol = min(max_ncol, len(image_files))
    nrow = (len(image_files) + ncol - 1) // ncol
    
    # 그리드 이미지 생성
    grid_width = width * ncol
    grid_height = height * nrow
    grid_image = Image.new("RGB", (grid_width, grid_height))
    draw = ImageDraw.Draw(grid_image)

    # 이미지 배치
    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        if should_resize:
            image = image.resize((width, height))
        x = (i % ncol) * width
        y = (i // ncol) * height
        grid_image.paste(image, (x, y))
        draw.rectangle((x, y, x + width, y + height), outline=(0, 0, 0), width=border_width)
    
    return grid_image

def create_composite_images(frames):
    """
    프레임들을 4x7 그리드 이미지로 구성하는 함수
    
    Args:
        frames: 프레임 목록
    Returns:
        list: 합성된 이미지 목록
    """
    reduced = skip_frames(frames, 280)
    composite_images = []

    for i in range(0, len(reduced), 28):
        frames_per_image = reduced[i:i+28]
        composite_image = create_grid_image(frames_per_image, 4)
        composite_images.append(composite_image)

    return composite_images

def group_frames_to_shots(frame_embeddings, min_similarity = 0.80):
    """
    유사한 프레임들을 샷으로 그룹화하는 함수
    
    Args:
        frame_embeddings: 프레임 임베딩 정보
        min_similarity: 최소 유사도 임계값
    Returns:
        list: 샷 정보 목록
    """
    shots = []
    current_shot = [frame_embeddings[0]]

    # 유사도 기반 프레임 그룹화
    for i in range(1, len(frame_embeddings)):
        prev = current_shot[-1]
        cur = frame_embeddings[i]
        similarity = embeddings.cosine_similarity(prev['embedding'], cur['embedding'])
        cur['similarity'] = similarity

        if similarity > min_similarity:
            current_shot.append(cur)
        else:
            shots.append(current_shot)
            current_shot = [cur]

    if current_shot:
        shots.append(current_shot)

    # 샷 정보 구성
    frames_in_shots = []
    for i, shot in enumerate(shots):
        frames_ids = [frame['frame_no'] for frame in shot]
        frames_in_shots.append({
            'shot_id': i,
            'frame_ids': frames_ids
        })

    return frames_in_shots


def plot_shots(frame_embeddings, num_shots):
    """
    샷 단위로 프레임을 시각화하는 함수
    
    Args:
        frame_embeddings: 프레임 임베딩 정보
        num_shots: 샷의 수
    """
    util.mkdir('shots')

    # 샷별로 프레임 그룹화
    shots = [[] for _ in range(num_shots)]
    for frame in frame_embeddings:
        shot_id = frame['shot_id']
        file = frame['file']
        shots[shot_id].append(file)

    # 각 샷 시각화
    for i in range(len(shots)):
        shot = shots[i]
        num_frames = len(shot)
        skipped_frames = skip_frames(shot)
        grid_image = create_grid_image(skipped_frames)
        
        # 이미지 크기 조정
        w, h = grid_image.size
        if h > 440:
            grid_image = grid_image.resize((w // 2, h // 2))
        w, h = grid_image.size
        
        # 결과 저장 및 표시
        print(f"Shot #{i:04d}: {num_frames} frames ({len(skipped_frames)} drawn) [{w}x{h}]")
        grid_image.save(f"shots/shot-{i:04d}.jpg")
        display(grid_image)
        grid_image.close()
    print('====')

def collect_similar_frames(frame_embeddings, frame_ids):
    """
    유사한 프레임들을 수집하는 함수
    
    Args:
        frame_embeddings: 프레임 임베딩 정보
        frame_ids: 프레임 ID 목록
    Returns:
        list: 유사한 프레임 ID 목록
    """
    similar_frames = []
    for frame_id in frame_ids:
        similar_frames_ids = [frame['idx'] for frame in frame_embeddings[frame_id]['similar_frames']]
        similar_frames.extend(similar_frames_ids)
    return sorted(list(set(similar_frames)))

def collect_related_shots(frame_embeddings, frame_ids):
    """
    관련된 샷들을 수집하는 함수
    
    Args:
        frame_embeddings: 프레임 임베딩 정보
        frame_ids: 프레임 ID 목록
    Returns:
        list: 관련된 샷 ID 목록
    """
    related_shots = []
    for frame_id in frame_ids:
        related_shots.append(frame_embeddings[frame_id]['shot_id'])
    return sorted(list(set(related_shots)))

def group_shots_in_scenes(frames_in_shots):
    """
    샷들을 씬으로 그룹화하는 함수
    
    Args:
        frames_in_shots: 샷별 프레임 정보
    Returns:
        list: 씬 정보 목록
    """
    # 씬 범위 계산
    scenes = [
        [
            min(frames_in_shot['related_shots']),
            max(frames_in_shot['related_shots']),
        ] for frames_in_shot in frames_in_shots
    ]

    # 씬 정렬 및 병합
    scenes = sorted(scenes, key=cmp_to_key(embeddings.cmp_min_max))
    stack = [scenes[0]]
    
    for i in range(1, len(scenes)):
        prev = stack[-1]
        cur = scenes[i]
        prev_min, prev_max = prev
        cur_min, cur_max = cur

        # 겹치는 씬 병합
        if cur_min >= prev_min and cur_min <= prev_max:
            new_scene = [
                min(cur_min, prev_min),
                max(cur_max, prev_max),
            ]
            stack.pop()
            stack.append(new_scene)
            continue
            
        stack.append(cur)

    # 씬 정보 구성
    return [{
        'scene_id': i,
        'shot_ids': stack[i],
    } for i in range(len(stack))]

def plot_scenes(frame_embeddings, num_scenes):
    """
    씬 단위로 프레임을 시각화하는 함수
    
    Args:
        frame_embeddings: 프레임 임베딩 정보
        num_scenes: 씬의 수
    """
    util.mkdir('scenes')

    # 씬별로 프레임 그룹화
    scenes = [[] for _ in range(num_scenes)]
    for frame in frame_embeddings:
        scene_id = frame['scene_id']
        file = frame['file']
        scenes[scene_id].append(file)

    # 각 씬 시각화
    for i in range(len(scenes)):
        scene = scenes[i]
        num_frames = len(scene)
        skipped_frames = skip_frames(scene)
        grid_image = create_grid_image(skipped_frames)
        
        # 이미지 크기 조정
        w, h = grid_image.size
        if h > 440:
            grid_image = grid_image.resize((w // 2, h // 2))
        w, h = grid_image.size
        
        # 결과 저장 및 표시
        print(f"Scene #{i:04d}: {num_frames} frames ({len(skipped_frames)} drawn) [{w}x{h}]")
        grid_image.save(f"scenes/scene-{i:04d}.jpg")
        display(grid_image)
        grid_image.close()
    print('====')

def make_chapter_item(chapter_id, scene_items, text = ''):
    """
    챕터 항목을 생성하는 함수
    
    Args:
        chapter_id: 챕터 ID
        scene_items: 씬 정보 목록
        text: 챕터 설명 텍스트
    Returns:
        dict: 챕터 정보
    """
    scene_ids = [scene['scene_id'] for scene in scene_items]
    return {
        'chapter_id': chapter_id,
        'scene_ids': [min(scene_ids), max(scene_ids)],
        'text': text,
    }

def group_scenes_in_chapters(conversations, shots_in_scenes, frames_in_shots):
    """
    씬들을 챕터로 그룹화하는 함수
    
    Args:
        conversations: 대화 정보
        shots_in_scenes: 씬별 샷 정보
        frames_in_shots: 샷별 프레임 정보
    Returns:
        list: 챕터 정보 목록
    """
    scenes = copy.deepcopy(shots_in_scenes)
    chapters = []

    # 대화 기반 챕터 구성
    for conversation in conversations['chapters']:
        start_ms = conversation['start_ms']
        end_ms = conversation['end_ms']
        text = conversation['reason']

        stack = []
        while len(scenes) > 0:
            scene = scenes[0]
            shot_min, shot_max = scene['shot_ids']
            frame_start = min(frames_in_shots[shot_min]['frame_ids']) * 1000
            frame_end = max(frames_in_shots[shot_max]['frame_ids']) * 1000

            # 대화 범위를 벗어난 경우
            if frame_start > end_ms:
                break

            # 대화 시작 전 씬들 처리
            if frame_end < start_ms:
                chapter = make_chapter_item(len(chapters), [scene])
                chapters.append(chapter)
                scenes.pop(0)
                continue

            stack.append(scene)
            scenes.pop(0)

        # 수집된 씬들로 챕터 생성
        if stack:
            chapter = make_chapter_item(len(chapters), stack, text)
            chapters.append(chapter)

    # 남은 씬들 처리
    for scene in scenes:
        chapter = make_chapter_item(len(chapters), [scene])
        chapters.append(chapter)

    return chapters

def plot_chapters(frame_embeddings, num_chapters):
    """
    챕터 단위로 프레임을 시각화하는 함수
    
    Args:
        frame_embeddings: 프레임 임베딩 정보
        num_chapters: 챕터의 수
    """
    try:
        os.mkdir('chapters')
    except Exception as e:
        print(e)

    # 챕터별로 프레임 그룹화
    chapters = [[] for _ in range(num_chapters)]
    for frame in frame_embeddings:
        chapter_id = frame['chapter_id']
        file = frame['file']
        chapters[chapter_id].append(file)

    # 각 챕터 시각화
    for i in range(len(chapters)):
        chapter = chapters[i]
        num_frames = len(chapter)
        skipped_frames = skip_frames(chapter)
        grid_image = create_grid_image(skipped_frames)
        
        # 이미지 크기 조정
        w, h = grid_image.size
        if h > 440:
            grid_image = grid_image.resize((w // 2, h // 2))
        w, h = grid_image.size
        
        # 결과 저장 및 표시
        print(f"Chapter #{i:04d}: {num_frames} frames ({len(skipped_frames)} drawn) [{w}x{h}]")
        grid_image.save(f"chapters/chapter-{i:04d}.jpg")
        display(grid_image)
        grid_image.close()
    print('====')