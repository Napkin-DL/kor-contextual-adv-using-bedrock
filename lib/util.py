import os
import json
import math
from fractions import Fraction
from shutil import rmtree

def save_to_file(output_file, data):
    """
    데이터를 파일로 저장하는 함수
    
    Args:
        output_file: 저장할 파일 경로
        data: 저장할 데이터 (문자열 또는 JSON 직렬화 가능한 객체)
    Returns:
        str: 저장된 파일 경로
    
    Example:
        >>> save_to_file('output.txt', 'Hello World')
        >>> save_to_file('data.json', {'key': 'value'})
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, ensure_ascii=False)
    return output_file

def save_json_to_file(name, json_data):
    """
    JSON 데이터를 파일로 저장하는 함수
    
    Args:
        name: 저장할 파일 경로
        json_data: 저장할 JSON 데이터
    Returns:
        dict: 저장된 JSON 데이터
    
    Example:
        >>> save_json_to_file('data.json', {'key': 'value'})
    """
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)
    return json_data

def to_hhmmssms(milliseconds):
    """
    밀리초를 HH:MM:SS.mmm 형식의 시간 문자열로 변환하는 함수
    
    Args:
        milliseconds: 밀리초 단위 시간
    Returns:
        str: HH:MM:SS.mmm 형식의 시간 문자열
    
    Example:
        >>> to_hhmmssms(3661000)
        '01:01:01.000'
    """
    # 시, 분, 초, 밀리초 계산
    hh = math.floor(milliseconds / 3600000)
    mm = math.floor((milliseconds % 3600000) / 60000)
    ss = math.floor((milliseconds % 60000) / 1000)
    ms = math.ceil(milliseconds % 1000)
    
    # 형식화된 문자열 반환
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def to_fraction(s):
    """
    문자열 또는 숫자를 분수로 변환하는 함수
    
    Args:
        s: 변환할 문자열(':' 구분자) 또는 숫자
    Returns:
        Fraction: 변환된 분수
    
    Example:
        >>> to_fraction('3:4')
        Fraction(3, 4)
        >>> to_fraction(0.75)
        Fraction(3, 4)
    """
    if isinstance(s, str):
        return Fraction(s.replace(':', '/')) 
    return Fraction(s)

def mkdir(directory):
    """
    디렉토리를 생성하는 함수
    
    Args:
        directory: 생성할 디렉토리 경로
    
    Example:
        >>> mkdir('data')
    """
    if not os.path.exists(directory):
        os.mkdir(directory)

def rmdir(directory):
    """
    디렉토리와 그 내용을 삭제하는 함수
    
    Args:
        directory: 삭제할 디렉토리 경로
    
    Example:
        >>> rmdir('data')
    """
    if os.path.exists(directory):
        rmtree(directory)