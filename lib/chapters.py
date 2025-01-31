import math
import webvtt
from functools import cmp_to_key
import json

def to_milliseconds(timestamp):
    """
    시간 문자열(HH:MM:SS.mmm)을 밀리초로 변환하는 함수
    
    Args:
        timestamp: 시간 문자열 (HH:MM:SS.mmm 형식)
    Returns:
        int: 밀리초 단위 시간
    """
    hh, mm, ss = timestamp.split(':')
    ss, ms = ss.split('.')
    hh, mm, ss, ms = map(int, (hh, mm, ss, ms))
    return (((hh * 3600) + (mm * 60) + ss) * 1000) + ms

def to_hhmmssms(milliseconds):
    """
    밀리초를 시간 문자열(HH:MM:SS.mmm)로 변환하는 함수
    
    Args:
        milliseconds: 밀리초 단위 시간
    Returns:
        str: HH:MM:SS.mmm 형식의 시간 문자열
    """
    hh = math.floor(milliseconds / 3600000)
    mm = math.floor((milliseconds % 3600000) / 60000)
    ss = math.floor((milliseconds % 60000) / 1000)
    ms = math.ceil(milliseconds % 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def parse_webvtt(file):
    """
    WebVTT 파일을 파싱하는 함수
    
    Args:
        file: WebVTT 파일 경로
    Returns:
        list: 자막 정보 리스트
    """
    captions = webvtt.read(file)
    captions = [{
        'text': caption.text,
        'start': caption.start,
        'end': caption.end,
        'start_ms': to_milliseconds(caption.start),
        'end_ms': to_milliseconds(caption.end)
    } for caption in captions]

    return captions

def cmp_timestamps(a, b):
    """
    타임스탬프 비교 함수
    
    Args:
        a, b: 비교할 챕터 정보
    Returns:
        int: 비교 결과 (-1, 0, 1)
    """
    if a['start_ms'] < b['start_ms']:
        return -1
    if a['start_ms'] > b['start_ms']:
        return 1
    return b['end_ms'] - a['end_ms']

def merge_chapters(chapters):
    """
    겹치는 챕터들을 병합하는 함수
    
    Args:
        chapters: 챕터 정보 리스트
    Returns:
        list: 병합된 챕터 리스트
    """
    # 타임스탬프를 밀리초로 변환
    for chapter in chapters:
        start = chapter['start']
        end = chapter['end']
        start_ms = to_milliseconds(start)
        end_ms = to_milliseconds(end)
        chapter['start_ms'] = start_ms
        chapter['end_ms'] = end_ms

    # 시작 시간 기준으로 정렬
    chapters = sorted(chapters, key=cmp_to_key(cmp_timestamps))

    # 겹치는 챕터 병합
    merged = [chapters[0]]
    for i in range(1, len(chapters)):
        prev = merged[-1]
        cur = chapters[i]

        prev_start_ms = prev['start_ms']
        prev_end_ms = prev['end_ms']
        cur_start_ms = cur['start_ms']
        cur_end_ms = cur['end_ms']

        # 유효성 검사
        if cur_end_ms < prev_start_ms:
            raise Exception('end_ms < start_ms? SHOULD NOT HAPPEN!')

        # 겹치지 않는 경우
        if cur_start_ms >= prev_end_ms:
            merged.append(cur)
            continue

        # 완전히 포함되는 경우
        if cur_start_ms > prev_start_ms and cur_end_ms < prev_end_ms:
            continue

        # 겹치는 경우 병합
        start_ms = min(prev_start_ms, cur_start_ms)
        end_ms = max(prev_end_ms, cur_end_ms)

        prev_duration = prev_end_ms - prev_start_ms
        cur_duration = cur_end_ms - cur_start_ms

        # 더 긴 챕터의 reason 사용
        reason = prev['reason']
        if cur_duration > prev_duration:
            reason = cur['reason']

        new_chapter = {
            'reason': reason,
            'start': to_hhmmssms(start_ms),
            'end': to_hhmmssms(end_ms),
            'start_ms': start_ms,
            'end_ms': end_ms
        }

        merged.pop()
        merged.append(new_chapter)

    return chapters

def validate_timestamps(chapters, captions):
    """
    챕터 타임스탬프를 자막 타임스탬프와 비교하여 검증하는 함수
    
    Args:
        chapters: 챕터 정보 리스트
        captions: 자막 정보 리스트
    Returns:
        list: 검증된 챕터 리스트
    """
    # 챕터별 자막 타임스탬프 수집
    for chapter in chapters:
        chapter_start = chapter['start_ms']
        chapter_end = chapter['end_ms']

        while len(captions) > 0:
            caption = captions[0]
            caption_start = caption['start_ms']
            caption_end = caption['end_ms']

            # 챕터 끝보다 자막이 늦게 시작하면 중단
            if caption_start >= chapter_end:
                break

            # 챕터 시작보다 자막이 일찍 끝나면 다음 자막으로
            if caption_end <= chapter_start:
                captions.pop(0)
                continue

            # 챕터 끝과 가까운 자막 선택
            if abs(chapter_end - caption_start) < abs(caption_end - chapter_end):
                break

            # 챕터에 자막 타임스탬프 추가
            if 'timestamps' not in chapter:
                chapter['timestamps'] = []
            chapter['timestamps'].append([caption_start, caption_end])

            captions.pop(0)

    # 챕터 경계 타임스탬프를 자막 타임스탬프에 맞춰 조정
    for chapter in chapters:
        if 'timestamps' not in chapter:
            continue
        
        chapter_start = chapter['start_ms']
        chapter_end = chapter['end_ms']

        caption_start = chapter['timestamps'][0][0]
        caption_end = chapter['timestamps'][-1][1]

        # 시작 시간 조정
        if chapter_start != caption_start:
            chapter['start_ms'] = caption_start
            chapter['start'] = to_hhmmssms(caption_start)

        # 종료 시간 조정
        if chapter_end != caption_end:
            chapter['end_ms'] = caption_end
            chapter['end'] = to_hhmmssms(caption_end)

        del chapter['timestamps']

    return chapters