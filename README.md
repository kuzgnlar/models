<img src="header_background.jpg" height ="38%" width="38%"></img> 

# Kuzgunlar Göreve Özgü Modeller

![GPL 3.0](https://img.shields.io/badge/license-GPLv3-red.svg)

Açık Hack Türkçe Doğal Dil İşleme Online Yarışma Programı kapsamında Kuzgunlar ekibi tarafından ince ayar(fine-tune) yapılan modellerin kullanımı ve eğitim süreci ile ilgili bilgileri içeren repodur.

Sunulan modellerin tümü [Stefan Schweter tarafından ön eğitimi yapılmış(pretrained) **ELECTRA base** modeli](https://github.com/stefan-it/turkish-bert/tree/master/electra) üzerine eğitilmiştir. *Electra base* modelinin seçilme sebebi [bu adreste](https://github.com/stefan-it/turkish-bert#pos-tagging) de belirtildiği gibi **PoS tagging** ve **NER** görevlerinde paylaşılmış diğer tüm Türkçe ön eğitimli modellere(BERTurk, DistilBERTurk, XLM-RoBERTa vb.) göre daha başarılı olmasıdır. Bu kapsamda **NER**, **Soru-Cevap** ve **Duygu Durumu Analizi** olmak üzere 3 model eğitilmiş, eğitilen tüm modeller [huggingface](https://huggingface.co/kuzgunlar) üzerinden paylaşılmıştır.

## NER
Kullanılan veri seti:

> [1] Sahin, H. Bahadir; Eren, Mustafa Tolga; Tirkaz, Caglar; Sonmez, Ozan; Yildiz, Eray (2017), “English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset”, Mendeley Data, v1 [http://dx.doi.org/10.17632/cdcztymf4k.1](http://dx.doi.org/10.17632/cdcztymf4k.1)

Örnek Uygulama: 

	from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
	
	model = AutoModelForTokenClassification.from_pretrained("kuzgunlar/electra-turkish-ner")
	tokenizer = AutoTokenizer.from_pretrained("kuzgunlar/electra-turkish-ner")
	ner=pipeline('ner', model=model, tokenizer=tokenizer)
	
	print(ner("Betelgeuse'un üstündeki yıldızlar avcının sağ kolunu Bellatrix'den ötede olan yıldızlarda avcının kalkanını oluşturur."))
	print(ner("Azerbaycanlı general ordusunu Tovuz'a konuşlandırdı."))
> [{'word': 'Bet', 'score': 0.9947249889373779, 'entity': 'B-space', 'index': 1}, {'word': '##el', 'score': 0.6537177562713623, 'entity': 'I-space', 'index': 2}, {'word': '##ge', 'score': 0.9661315679550171, 'entity': 'I-space', 'index': 3}, {'word': '##use', 'score': 0.8851485848426819, 'entity': 'I-space', 'index': 4}, {'word': "'", 'score': 0.5226207375526428, 'entity': 'I-space', 'index': 5}, {'word': 'yıldızlar', 'score': 0.8456777930259705, 'entity': 'B-space', 'index': 8}, {'word': '##rix', 'score': 0.5831472277641296, 'entity': 'B-space', 'index': 15}, {'word': 'yıldızlar', 'score': 0.9033623337745667, 'entity': 'B-space', 'index': 21}]
> [{'word': 'Azerbaycan', 'score': 0.7510120272636414, 'entity': 'B-location', 'index': 1}, {'word': 'general', 'score': 0.999763011932373, 'entity': 'B-military', 'index': 3}, {'word': 'To', 'score': 0.8212651610374451, 'entity': 'B-location', 'index': 6}, {'word': '##v', 'score': 0.6104686260223389, 'entity': 'I-location', 'index': 7}]

## Soru-Cevap

Kullanılan veri seti:

> [1] [TQUAD](https://github.com/TQuad/turkish-nlp-qa-dataset)
> [2] [Kuzgunlar question-answer dataset](https://github.com/kuzgnlar/datasets/tree/master/question-answer)

Örnek Uygulama: 

	from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

	model = AutoModelForQuestionAnswering.from_pretrained("kuzgunlar/electra-turkish-qa")
	tokenizer = AutoTokenizer.from_pretrained("kuzgunlar/electra-turkish-qa")
	qna=pipeline("question-answering", model=model, tokenizer=tokenizer)

	context = "NLP yani Doğal Dil İşleme, doğal dillerin kurallı yapısının çözümlenerek anlaşılması veya yeniden üretilmesi amacını taşır.Bu çözümlemenin insana getireceği kolaylıklar, yazılı dokümanların otomatik çevrilmesi, soru-cevap makineleri, otomatik konuşma ve komut anlama, konuşma sentezi, konuşma üretme, otomatik metin özetleme, bilgi sağlama gibi birçok başlıkla özetlenebilir. Bilgisayar teknolojisinin yaygın kullanımı, bu başlıklardan üretilen uzman yazılımların gündelik hayatımızın her alanına girmesini sağlamıştır. Örneğin, tüm kelime işlem yazılımları birer imlâ düzeltme aracı taşır. Bu araçlar aslında yazılan metni çözümleyerek dil kurallarını denetleyen doğal dil işleme yazılımlarıdır. \n Batı dillerinde SAPI (Microsoft şirketinin konuşma sentezleyici üretmek amacı ile satışa sunduğu geliştirici program) tabanlı Konuşma sentezleyici bileşenleri, yazılımcıların multimedia (çoklu ortam) sunuları hazırlamaları için hizmete sunulmuştur. \n Konuşma ve komut anlama yazılımları ise gelecekte insan ve bilgisayar arasındaki klavye, fare gibi veri girişi aygıtlarını ortadan kaldıracak yazılımlardır. Bu gelişmeler makine-insan iletişiminde yeni ve devrimci değişimlere yol açacak ve bilgisayarların daha çok insan tarafından kabul görmesine yol açacaktır."
	print(qna(question="NLP'nin amacı nedir?", context=context))
	print(qna(question="Konuşma ve komut anlama yazılımları neyi ortadan kaldıracaktır?", context=context))

> {'score': 0.8954808187348605, 'start': 27, 'end': 108, 'answer': 'doğal dillerin kurallı yapısının çözümlenerek anlaşılması veya yeniden üretilmesi'}
> {'score': 0.5904039331509782, 'start': 1000, 'end': 1072, 'answer': 'insan ve bilgisayar arasındaki klavye, fare gibi veri girişi aygıtlarını'}
	

## Duygu Durumu Analizi

Kullanılan veri seti:

> [1] Ucan A, Naderalvojoud B, Sezer EA, Sever H. SentiWordNet for new language: automatic translation approach.In: 12th International Conference on Signal-Image Technology & Internet-Based Systems; Naples, Italy; 2016. pp.308-315.

Örnek Uygulama: 

	from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
	
	model = AutoModelForSequenceClassification.from_pretrained("kuzgunlar/electra-turkish-sentiment-analysis")
	tokenizer = AutoTokenizer.from_pretrained("kuzgunlar/electra-turkish-sentiment-analysis")
	sa= pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

	print(sa("Malesef günümüzde tüm dünyada reklam işten daha önemli."))
	print(sa("Biz umudumuzu hiç kaybetmedik."))
	
> [{'label': 'negative', 'score': 0.5602248311042786}]
> [{'label': 'positive', 'score': 0.5182426571846008}]
