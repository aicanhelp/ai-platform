package hermesrouter

import "time"

const (
	DefaultInferenceModel       = "Qwen/Qwen3-8B"
	DefaultInferenceMessage     = "Hello! How are you today"
	DefaultInferenceLongMessage = "Hello! I hope this message finds you well. Today, I wanted to discuss a variety of topics ranging from technology and science to culture and daily life. Let's start with artificial intelligence: recent advancements in machine learning have revolutionized industries like healthcare, finance, and transportation. For example, AI-powered diagnostic tools can now detect diseases with greater accuracy than ever before, while self-driving cars continue to improve safety on the roads. Moving on to science, breakthroughs in quantum computing hold the promise of solving complex problems that are beyond the reach of classical computers. This could lead to innovations in drug discovery, climate modeling, and cryptography. On the cultural front, the rise of streaming services has transformed how we consume media, making movies, music, and television shows more accessible globally. However, this shift has also raised questions about copyright, artistic integrity, and the impact on traditional media outlets. In daily life, smartphones and smart home devices have become integral parts of our routines, connecting us to information and each other in ways that were unimaginable a decade ago. Yet, there are concerns about screen time, digital privacy, and the potential for technology to isolate people socially. Additionally, I'm curious about your thoughts on sustainable living and how individuals can reduce their environmental footprint. Simple changes like recycling, using renewable energy sources, and reducing single-use plastics can make a significant difference when adopted widely. Another topic that interests me is education: how can we adapt teaching methods to prepare students for a rapidly changing world? Online learning platforms have expanded access to education, but they also present challenges such as maintaining student engagement and ensuring equitable access to technology. Finally, I'd like to hear your perspective on creativity and innovation. What role do you think artificial intelligence will play in fostering creativity, and how can we balance technological progress with human expression? Thank you for taking the time to consider these questions; I look forward to your insights."
	DefaultInferenceStream      = false
	DefaultMaxTokens            = 512
	containerReadyTimeout       = 10 * time.Minute
)

const (
	AggregateBaseValuesPath = "hermes-router-config/values.yaml"
	PDBaseValuesPath        = "hermes-router-config/values-pd-base.yaml"
)

// EPP 启动失败时输出的关键错误短语，供部署门禁负向用例断言使用。
const (
	errPredictorAddressMustUnix = "predictorAddress must use unix://"
	errScoreWeightsSumGT0       = "ttftWeight and tpotWeight must sum to greater than 0"
	errFallbackWeightsSumGT0    = "kvWeight, queueWeight, prefixWeight, and inflightWeight must sum to greater than 0"
)
