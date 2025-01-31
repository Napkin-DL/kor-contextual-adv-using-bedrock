import json
import boto3
import json_repair
from termcolor import colored
from lib import frames

# Claude 모델 ID와 버전 정의
HAIKU_MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0'
SONNET_MODEL_ID = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
MODEL_VER = 'bedrock-2023-05-31'

# 모델별 가격 설정 (입력 토큰당, 출력 토큰당)
HAIKU_CLAUDE_PRICING = (0.00025, 0.00125)
SONNET_CLAUDE_PRICING = (0.003, 0.015)

def analyze_conversations(model_flag, transcript_file, analysis_df):
    """
    대화 내용을 분석하는 함수
    
    Args:
        model_flag: 사용할 모델 타입 ("SONNET" 또는 "HAIKU")
        transcript_file: 트랜스크립트 파일 경로
        analysis_df: 분석 데이터프레임
    Returns:
        tuple: (분석 결과, 사용된 모델 ID)
    """
    # 모델 선택
    model_id = SONNET_MODEL_ID if model_flag == "SONNET" else HAIKU_MODEL_ID
    
    # 메시지 구성
    messages = []
    messages.append(make_transcript(transcript_file))
    messages.append({'role': 'assistant', 'content': 'Got the transcript. What output format?'})
    messages.append(make_conversation_example())
    messages.append({'role': 'assistant', 'content': '{'})
    messages.append(make_additional_info(analysis_df))
    messages.append({'role': 'assistant', 'content': '{'})

    # 시스템 프롬프트 설정
    system = 'You are a media operation assistant who analyses movie transcripts in WebVTT format and suggest chapter points based on the topic changes in the conversations. It is important to read the entire transcripts.'

    # 모델 파라미터 설정
    model_params = {
        'anthropic_version': MODEL_VER,
        'max_tokens': 4096,
        'temperature': 0.0,
        'top_p': 0.7,
        'top_k': 200,
        'stop_sequences': ['\n\nHuman:'],
        'system': system,
        'messages': messages
    }

    # 추론 실행
    try:
        response = inference(model_id, model_params)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response = inference(model_params)
    
    return response, model_id

def get_contextual_information(model_flag, images, text, iab_definitions):
    """
    컨텍스트 정보를 분석하는 함수
    
    Args:
        model_flag: 사용할 모델 타입
        images: 이미지 리스트
        text: 대화 텍스트
        iab_definitions: IAB 분류 정의
    Returns:
        tuple: (분석 결과, 사용된 모델 ID)
    """
    # 모델 선택
    model_id = SONNET_MODEL_ID if model_flag == "SONNET" else HAIKU_MODEL_ID
    
    # 작업 설명 정의
    task_all = 'You are asked to provide the following information: a detail description to describe the scene, identify the most relevant IAB taxonomy, GARM, sentiment, and brands and logos that may appear in the scene, and five most relevant tags from the scene.'
    task_iab_only = 'You are asked to identify the most relevant IAB taxonomy.'
    system = 'You are a media operation engineer. Your job is to review a portion of a video content presented in a sequence of consecutive images. Each image also contains a sequence of frames presented in a 4x7 grid reading from left to right and then from top to bottom. You may also optionally be given the conversation of the scene that helps you to understand the context. {0} It is important to return the results in JSON format and also includes a confidence score from 0 to 100. Skip any explanation.Answer in Korean.';

    # 메시지 구성
    messages = []
    messages.append(make_image_message(images))
    messages.append({'role': 'assistant', 'content': 'Got the images. Do you have the conversation of the scene?'})
    messages.append(make_conversation_message(text))
    messages.append({'role': 'assistant', 'content': 'OK. Do you have other information to provide?'})
    
    # 추가 정보 구성
    other_information = [
        make_iab_taxonomoies(iab_definitions['tier1']),
        make_garm_taxonomoies(),
        make_sentiments()
    ]
    messages.append({'role': 'user', 'content': other_information})
    
    # 출력 형식 추가
    messages.append({'role': 'assistant', 'content': 'OK. What output format?'})
    messages.append(make_output_example())
    messages.append({'role': 'assistant', 'content': '{'})

    # 모델 파라미터 설정
    model_params = {
        'anthropic_version': MODEL_VER,
        'max_tokens': 4096,
        'temperature': 0.1,
        'top_p': 0.7,
        'top_k': 20,
        'stop_sequences': ['\n\nHuman:'],
        'system': system.format(task_all),
        'messages': messages
    }

    # 추론 실행
    try:
        response = inference(model_id, model_params)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response = inference(model_params)

    return response, model_id

def inference(model_id, model_params):
    """
    AWS Bedrock을 사용하여 모델 추론을 실행하는 함수
    
    Args:
        model_id: 사용할 모델 ID
        model_params: 모델 파라미터
    Returns:
        dict: 추론 결과
    """
    # API 설정
    accept = 'application/json'
    content_type = 'application/json'

    # Bedrock 클라이언트 생성
    bedrock_runtime_client = boto3.client(service_name='bedrock-runtime')

    # 모델 호출
    response = bedrock_runtime_client.invoke_model(
        body=json.dumps(model_params),
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    # 응답 처리
    response_body = json.loads(response.get('body').read())
    response_content = response_body['content'][0]['text']
    
    # JSON 형식 검증 및 수정
    if response_content[0] != '{':
        response_content = '{' + response_content

    try:
        response_content = json.loads(response_content)
    except Exception as e:
        print(colored("Malformed JSON response. Try to repair it...", 'red'))
        try:
            response_content = json_repair.loads(response_content)
        except Exception as e:
            print(colored("Failed to repair the JSON response...", 'red'))
            print(colored(response_content, 'red'))
            raise e

    response_body['content'][0]['json'] = response_content
    return response_body

def display_conversation_cost(response, model_id):
    """
    대화 분석 비용을 계산하고 표시하는 함수
    
    Args:
        response: 모델 응답
        model_id: 사용된 모델 ID
    Returns:
        dict: 비용 정보
    """
    # 모델별 가격 설정
    if model_id == HAIKU_MODEL_ID:
        input_per_1k, output_per_1k = HAIKU_CLAUDE_PRICING
    elif model_id == SONNET_MODEL_ID:
        input_per_1k, output_per_1k = SONNET_CLAUDE_PRICING

    # 토큰 수 계산
    input_tokens = response['usage']['input_tokens']
    output_tokens = response['usage']['output_tokens']

    # 비용 계산
    conversation_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    # 결과 출력
    print('\n')
    print('========================================================================')
    print('Estimated cost:', colored(f"${conversation_cost}", 'green'), 
          f"in us-west-2 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
    print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': conversation_cost,
    }

def display_contextual_cost(usage, model_id):
    """
    컨텍스트 분석 비용을 계산하고 표시하는 함수
    
    Args:
        usage: 사용량 정보
        model_id: 사용된 모델 ID
    Returns:
        dict: 비용 정보
    """
    # 모델별 가격 설정
    if model_id == HAIKU_MODEL_ID:
        input_per_1k, output_per_1k = HAIKU_CLAUDE_PRICING
    elif model_id == SONNET_MODEL_ID:
        input_per_1k, output_per_1k = SONNET_CLAUDE_PRICING

    # 토큰 수 계산
    input_tokens = usage['input_tokens']
    output_tokens = usage['output_tokens']

    # 비용 계산
    contextual_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    # 결과 출력
    print('\n')
    print('========================================================================')
    print('Estimated cost:', colored(f"${round(contextual_cost, 4)}", 'green'), 
          f"in us-west-2 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
    print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': contextual_cost,
    }

def make_conversation_example():
    """대화 분석 결과의 예시 형식을 생성하는 함수"""
    example = {
        'chapters': [
            {
                'start': '00:00:10.000',
                'end': '00:00:32.000',
                'reason': 'It appears the chapter talks about...'
            }
        ]
    }

    return {
        'role': 'user',
        'content': 'JSON format. An example of the output:\n{0}\n'.format(json.dumps(example))
    }

def make_transcript(transcript_file):
    """트랜스크립트 파일을 읽어 메시지 형식으로 변환하는 함수"""
    with open(transcript_file, encoding="utf-8") as f:
        transcript = f.read()

    return {
        'role': 'user',
        'content': 'Here is the transcripts in <transcript> tag:\n<transcript>{0}\n</transcript>\n'.format(transcript)
    }

def make_additional_info(analysis_df):
    """추가 분석 정보를 메시지 형식으로 변환하는 함수"""
    return {
        'role': 'user',
        'content': 'After identifying the chapter points from the transcript, you may reference the Break Scores from <table> tag only to validate if your suggested chapter points occur at suitable moments in the video.\n<table>{0}\n</table>\nAnswer in Korean.'.format(analysis_df)
    }

def make_image_message(images):
    """이미지 시퀀스를 메시지 형식으로 변환하는 함수"""
    image_contents = [{
        'type': 'text',
        'text': 'Here are {0} images containing frame sequence that describes a scene.'.format(len(images))
    }]

    for image in images:
        bas64_image = frames.image_to_base64(image)
        image_contents.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/jpeg',
                'data': bas64_image
            }
        })

    return {
        'role': 'user',
        'content': image_contents
    }

def make_conversation_message(text):
    """대화 내용을 메시지 형식으로 변환하는 함수"""
    message = {
        'role': 'user',
        'content': 'No conversation.'
    }

    if text:
        message['content'] = 'Here is the conversation of the scene in <conversation> tag.\n<conversation>\n{0}\n</conversation>\n'.format(text)

    return message

def make_iab_taxonomoies(iab_list):
    """IAB 분류 목록을 메시지 형식으로 변환하는 함수"""
    iab = [item['name'] for item in iab_list]
    iab.append('None')

    return {
        'type': 'text',
        'text': 'Here is a list of IAB Taxonomies in <iab> tag:\n<iab>\n${0}\n</iab>\nOnly answer the IAB taxonomy from this list.'.format('\n'.join(iab))
    }

def make_garm_taxonomoies():
    """
    GARM 분류 목록을 메시지 형식으로 변환하는 함수
    
    Returns:
        dict: GARM 분류 메시지
    """
    garm = [
        'Adult & Explicit Sexual Content',
        'Arms & Ammunition',
        'Crime & Harmful acts to individuals and Society, Human Right Violations',
        'Death, Injury or Military Conflict',
        'Online piracy',
        'Hate speech & acts of aggression',
        'Obscenity and Profanity, including language, gestures, and explicitly gory, graphic or repulsive content intended to shock and disgust',
        'Illegal Drugs, Tobacco, ecigarettes, Vaping, or Alcohol',
        'Spam or Harmful Content',
        'Terrorism',
        'Debated Sensitive Social Issue',
        'None',
    ]

    return {
        'type': 'text',
        'text': 'Here is a list of GARM Taxonomies in <garm> tag:\n<garm>\n{0}\n</garm>\nOnly answer the GARM taxonomy from this list.'.format('\n'.join(garm))
    }

def make_sentiments():
    """
    감정 분류 목록을 메시지 형식으로 변환하는 함수
    
    Returns:
        dict: 감정 분류 메시지
    """
    sentiments = ['Positive', 'Neutral', 'Negative', 'None']

    return {
        'type': 'text',
        'text': 'Here is a list of Sentiments in <sentiment> tag:\n<sentiment>\n{0}\n</sentiment>\nOnly answer the sentiment from this list.'.format('\n'.join(sentiments))
    }

def make_output_example():
    """
    출력 형식의 예시를 생성하는 함수
    
    Returns:
        dict: 출력 형식 예시 메시지
    """
    example = {
        'description': {
            'text': 'The scene describes...',
            'score': 98
        },
        'sentiment': {
            'text': 'Positive',
            'score': 90
        },
        'iab_taxonomy': {
            'text': 'Station Wagon',
            'score': 80
        },
        'garm_taxonomy': {
            'text': 'Online piracy',
            'score': 90
        },
        'brands_and_logos': [
            {
                'text': 'Amazon',
                'score': 95
            },
            {
                'text': 'Nike',
                'score': 85
            }
        ],
        'relevant_tags': [
            {
                'text': 'BMW',
                'score': 95
            }
        ]            
    }
    
    return {
        'role': 'user',
        'content': 'Return JSON format. An example of the output:\n{0}\n'.format(json.dumps(example))
    }