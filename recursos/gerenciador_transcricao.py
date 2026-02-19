#import whisper
from recursos import configuracoes
import os
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pyannote.audio import Pipeline
import torch
import librosa

def transcrever(local_arquivo_audio):
    try:
        if not os.path.isfile(local_arquivo_audio):
            raise FileNotFoundError(f"Arquivo não encontrado: {local_arquivo_audio}")

        # Carrega o modelo base (pode usar 'tiny', 'base', 'small', 'medium', 'large')
        model = whisper.load_model("large-v3")
        
        # Transcreve o áudio
        result = model.transcribe(local_arquivo_audio, language="pt", fp16=False)

        
        nome_sem_extensao = os.path.splitext(os.path.basename(local_arquivo_audio))[0]
        
        local_arquivo_texto = f'{configuracoes.LOCAL_ARQUIVOS_TRANSCRITOS}{nome_sem_extensao}.txt'

  
        
        with open(local_arquivo_texto, "w", encoding="utf-8") as f:
            for segmento in result['segments']:
                texto = segmento['text'].strip()
                f.write(texto + "\n")  # Dupla quebra para deixar espaçado
                
        print(f'✅ Transcrição concluída. Confira o arquivo "{nome_sem_extensao}.txt".')
    except Exception as e:
        print(f"Erro ao transcrever o áudio: {e}")
        return None


def transcrever_gpu(local_arquivo_audio):
    try:
        if not os.path.isfile(local_arquivo_audio):
            raise FileNotFoundError(f"Arquivo não encontrado: {local_arquivo_audio}")

        print('Carrega o modelo base')
        model = WhisperModel(
            "large-v3",
            device="cuda",  # ou "cpu"
            compute_type="float16"  # ou "int8" para economia de memória
        )

        print('Transcreve o áudio')
        segments, info = model.transcribe(local_arquivo_audio, language="pt")

        nome_sem_extensao = os.path.splitext(os.path.basename(local_arquivo_audio))[0]

        local_arquivo_texto = f'{configuracoes.LOCAL_ARQUIVOS_TRANSCRITOS}{nome_sem_extensao}.txt'

        print('Escreve o arquivo')
        with open(local_arquivo_texto, "w", encoding="utf-8") as f:
            for segment in segments:
                texto = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                f.write(texto + "\n")  # Dupla quebra para deixar espaçado

        print(f'✅ Transcrição concluída. Confira o arquivo "{nome_sem_extensao}.txt".')
    except Exception as e:
        print(f"Erro ao transcrever o áudio: {e}")
        return None


def transcrever_com_falantes(local_arquivo_audio, whisper_model="base", device="cuda"):
    try:
        # 1. Carregar modelos
        print("Carregando modelos...")

        # Whisper para transcrição
        whisper = WhisperModel(whisper_model, device=device, compute_type="int8")

        # pyannote para diarização (identificação de falantes)
        # Você precisa aceitar os termos em: https://huggingface.co/pyannote/speaker-diarization
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"  # Opcional se modelo for público
        )

        # 2. Identificar falantes
        print("Identificando falantes...")
        diarization = diarization_pipeline(local_arquivo_audio)

        # 3. Transcrever áudio
        print("Transcrevendo áudio...")
        segments, info = whisper.transcribe(local_arquivo_audio, language="pt", beam_size=5)

        # 4. Combinar transcrição com falantes
        print("Combinando transcrição com falantes...")

        # Converter segmentos do whisper para lista
        transcription_segments = []
        for segment in segments:
            transcription_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })

        # Combinar com informações de falantes
        resultado_final = []

        for trans_seg in transcription_segments:
            # Encontrar qual falante estava ativo no meio do segmento
            mid_time = (trans_seg['start'] + trans_seg['end']) / 2

            speaker = "Desconhecido"
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    speaker = speaker_label
                    break

            resultado_final.append({
                'start': trans_seg['start'],
                'end': trans_seg['end'],
                'speaker': speaker,
                'text': trans_seg['text']
            })

        print('Escreve o arquivo')
        nome_sem_extensao = os.path.splitext(os.path.basename(local_arquivo_audio))[0]
        local_arquivo_texto = f'{configuracoes.LOCAL_ARQUIVOS_TRANSCRITOS}{nome_sem_extensao}.txt'


        with open(local_arquivo_texto, "w", encoding="utf-8") as f:
            for result in resultado_final:
                texto = f"[{result['start']}s:{result['end']}s]->[{result['speaker']}]: {result['text']}"
                f.write(texto + "\n")  # Dupla quebra para deixar espaçado

        print('FIM - Escreve o arquivo')
    except Exception as e:
        print(f"Erro ao transcrever o áudio: {e}")
        return None

def transcrever_batch(local_arquivo_audio, whisper_model="base", device="cuda"):

    """
    Transcreve áudio identificando diferentes falantes
    """
    import gc





    # 1. Carregar modelos
    print("Carregando modelos...")

    # Whisper para transcrição
    model = WhisperModel(whisper_model, device=device, compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)
    # pyannote para diarização (identificação de falantes)
    # Você precisa aceitar os termos em: https://huggingface.co/pyannote/speaker-diarization
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"  # Opcional se modelo for público
    )

    # 2. Identificar falantes
    print("Identificando falantes...")
    diarization = diarization_pipeline(local_arquivo_audio)

    # 3. Transcrever áudio
    print("Transcrevendo áudio...")
    segments, info = batched_model.transcribe(local_arquivo_audio, language="pt", beam_size=5, batch_size=4)

    # 4. Combinar transcrição com falantes
    print("Combinando transcrição com falantes...")

    gc.collect()

    torch.cuda.empty_cache()

    # Converter segmentos do whisper para lista
    transcription_segments = []
    for segment in segments:
        transcription_segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        })

    # Combinar com informações de falantes
    resultado_final = []
    print("Identificando  com falantes...")
    for trans_seg in transcription_segments:
        # Encontrar qual falante estava ativo no meio do segmento
        mid_time = (trans_seg['start'] + trans_seg['end']) / 2

        speaker = "Desconhecido"
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if turn.start <= mid_time <= turn.end:
                speaker = speaker_label
                break

        resultado_final.append({
            'start': trans_seg['start'],
            'end': trans_seg['end'],
            'speaker': speaker,
            'text': trans_seg['text']
        })

    print('Escreve o arquivo')
    nome_sem_extensao = os.path.splitext(os.path.basename(local_arquivo_audio))[0]
    local_arquivo_texto = f'{configuracoes.LOCAL_ARQUIVOS_TRANSCRITOS}{nome_sem_extensao}.txt'


    with open(local_arquivo_texto, "w", encoding="utf-8") as f:
        for result in resultado_final:
            texto = f"[{result['start']}s:{result['end']}s]->[{result['speaker']}]: {result['text']}"
            f.write(texto + "\n")  # Dupla quebra para deixar espaçado