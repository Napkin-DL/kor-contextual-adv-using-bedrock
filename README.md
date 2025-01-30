# kor-contextual-adv-using-bedrock

이 프로젝트는 https://github.com/aws-samples/contextual-advertising-using-generative-ai-on-aws.git 를 기반으로 번역한 github 입니다.

Context advertising은 사용자가 소비하는 웹페이지나 미디어의 context에 맞춰 광고를 매칭하는 형태의 타겟 광고입니다. 이 과정에는 게시자(웹사이트 또는 콘텐츠 소유자), 광고주, 소비자라는 세 주체가 관여합니다. 게시자는 플랫폼과 콘텐츠를 제공하고, 광고주는 context에 맞는 광고를 제작합니다. 소비자는 콘텐츠와 상호작용하며, context에 기반한 관련 광고가 표시되어 더욱 개인화되고 관련성 높은 광고 경험을 만들어냅니다.

VOD 플랫폼 스트리밍용 미디어 콘텐츠에 광고를 삽입하는 것은 context advertising의 도전적인 영역입니다. 이 과정은 전통적으로 전문가가 콘텐츠를 분석하고 관련 키워드나 카테고리를 할당하는 수동 태깅에 의존했습니다. 하지만 이러한 접근 방식은 시간이 많이 소요되고 주관적이며, 콘텐츠의 전체 context나 뉘앑스를 포착하지 못할 수 있습니다. 전통적인 AI/ML 솔루션으로 이 과정을 자동화할 수 있지만, 대개 방대한 학습 데이터가 필요하고 비용이 많이 들며 기능이 제한적입니다.

대규모 언어 모델이 지원하는 생성형 AI는 이러한 과제에 대한 유망한 해결책을 제시합니다. 이러한 모델의 방대한 지식과 context 이해를 활용하여 방송사와 콘텐츠 제작자는 미디어 자산에 대한 contextual 인사이트와 분류 체계를 자동으로 생성할 수 있습니다. 이러한 접근 방식은 프로세스를 간소화하고 정확하고 포괄적인 context 이해를 제공하여 효과적인 광고 타겟팅과 미디어 아카이브의 수익화를 가능하게 합니다.

이 프로젝트에서는 Media2Cloud Guidance V4의 새로운 기능 중 하나인 장면 및 광고 브레이크 감지와 광고 브레이크의 contextual 이해에 대해 자세히 살펴볼 것입니다. AWS에서 생성형 AI를 사용하여 광고를 위한 contextual 관련 인사이트와 분류 체계를 만드는 방법을 단계별로 시연할 것입니다. 이를 통해 방송사와 콘텐츠 제작자는 미디어 자산을 더 효과적으로 수익화하고 미디어 아카이브에서 더 큰 가치를 추출할 수 있습니다. 생성형 AI의 힘을 활용함으로써 새로운 수익원을 창출하고 시청자에게 더욱 개인화되고 매력적인 광고 경험을 제공할 수 있습니다.

## Prerequisites to run the sample notebook

- AWS 계정이 필요합니다. AWS ID에 Amazon Bedrock, Amazon SageMaker 및 파일 업로드를 위한 Amazon S3 액세스 권한을 포함한 필요한 권한이 있는지 확인하세요.
- Amazon Bedrock에서 모델 액세스를 관리할 수 있는 권한이 필요합니다. 이 솔루션에는 Claude 3 Sonnet 및 Claude 3 Haiku 모델이 필요합니다.
- 이 노트북은 Amazon SageMaker Studio의 기본 Python3 커널을 사용하여 테스트되었습니다. ml.m5.2xlarge CPU 인스턴스가 권장됩니다. Amazon SageMaker Studio용 도메인 설정에 대한 문서를 참조하세요.
- 노트북을 실행하려면 ffmpeg, open-cv, webvtt-py와 같은 서드파티 라이브러리가 필요합니다. 코드 섹션을 실행하기 전에 지침을 따라 먼저 설치하세요.
- 예제 비디오는 Creative Commons Attribution 4.0 International Public License에 따라 Netflix Open Content에서 다운로드한 단편 영화인 Meridian입니다.

## Notebook configuration

이 샘플의 노트북은 다음과 같은 SageMaker Studio 구성으로 잘 실행됩니다:

- Image: Data Science 3.0
- Instance Type: ml.t3.medium (권장)
- Python version: 3.10

## License

[LICENSE](./LICENSE) 파일을 참조하세요.