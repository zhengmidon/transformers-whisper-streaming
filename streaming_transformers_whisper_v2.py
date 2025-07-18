from transformers import pipeline,GenerationConfig,AutoProcessor,AutoModelForSpeechSeq2Seq
from transformers import GenerationConfig
import json
import torch
import time
from dataclasses import dataclass
import torchaudio

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 128
HOP_LENGTH = 160
CHUNK_LENGTH = 30

@dataclass
class StreamingConfig:
    chunk_length: float = 1.0
    cache_length: int = 10
    model_path: str = 'models/whisper-large-v3'
    langid: str = 'en'
    tolerance: int = 2

class Segment:
    def __init__(self, audio_path, samples_to_read, samples_in_chunk):
        self.audio_path = audio_path
        audio, sr = torchaudio.load(audio_path, normalize=True)
        self.audio = audio.squeeze()
        self.audio_len_s = self.audio.shape[0] / SAMPLE_RATE
        assert sr == SAMPLE_RATE
        self.samples_to_read = samples_to_read
        self.samples_in_chunk = samples_in_chunk
        self.buffer_len = samples_in_chunk - samples_to_read

    def __iter__(self):
        frames_in_chunk = self.audio[:self.samples_in_chunk]
        read_pointer = frames_in_chunk.shape[0]
        yield frames_in_chunk, (read_pointer >= self.audio.shape[0])
        while read_pointer < self.audio.shape[0]:
            frames_in_chunk = torch.cat(
                (frames_in_chunk[-self.buffer_len:], self.audio[read_pointer:read_pointer+self.samples_to_read]),
                dim=0
                )
            read_pointer += self.samples_to_read
            yield frames_in_chunk, (read_pointer >= self.audio.shape[0])


class SegmentWrapper(Segment):
    def __init__(self, audio_path, segment_length):
        frames_to_read = int((segment_length * SAMPLE_RATE) / HOP_LENGTH)
        samples_to_read = frames_to_read * HOP_LENGTH
        samples_in_chunk = samples_to_read + N_FFT - HOP_LENGTH
        super().__init__(
            audio_path, 
            samples_to_read=samples_to_read, 
            samples_in_chunk=samples_in_chunk)

class StreamingWhisperTransformers:
    def __init__(self, cfg):
        self.config = cfg
        # set device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # data type used when inferring
        self.infer_dtype = torch.float16

        # load processor
        self.processor = AutoProcessor.from_pretrained(cfg.model_path, language=cfg.langid, task="transcribe")

        # load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.model_path, low_cpu_mem_usage=True, torch_dtype=self.infer_dtype
        ).to(self.device)

        # chunk length, in second
        self.chunk_length = cfg.chunk_length

        # chunk cache
        self.chunk_cache = []
        self.cache_length = cfg.cache_length

        # initial tokens
        self.init_tokens = torch.tensor([self.processor.tokenizer.prefix_tokens]).to(self.device)

        # first chunk in a workflow
        self.first_chunk = True

        # the new tokens in the last prediction
        self.last_tokens = None 

        # current trust token length
        self.curr_token_length = 4 # [<|startoftranscript|><|langid|><|transcribe|><|notimestamps|>]

        # new generated tokens of this cache
        self.trust_tokens = []

        # token timestamps for segment refreshment
        self.token_timestamps = None
        self.cur_segment_length = 0

        # must output after empty_out exceeds tolerance
        self.empty_out = 0
        self.tolerance = cfg.tolerance


    def refresh_segment(self, complete=False):
        if not complete and len(self.chunk_cache) > 2:
            # retain part of cache audio for better context
            chunk, tokens = self.seek_timestamp()
            # renew tokens
            self.curr_token_length = 4 + tokens.size(0)
            self.trust_tokens = [tokens.unsqueeze(0)]
            self.chunk_cache = [chunk]
        else:
            # record last context tokens
            self.curr_token_length = 4
            self.trust_tokens = []
            self.chunk_cache = []
    
    def set_infer_configs(self, inputs):
        configs = {"use_cache": True, 
                    'num_frames': inputs["num_frames"], 
                    "repetition_penalty": 1.2,
                    }
        
        # set inferring configs
        generation_config, model_kargs = self.model._prepare_generation_config(generation_config=None, **configs)
        return_dict_in_generate = self.model._set_return_outputs(
            return_dict_in_generate=None,
            return_token_timestamps=True, # set true to get token-level timestamps based on DTW
            logprob_threshold=None,
            generation_config=generation_config,
        )
        timestamp_begin = self.model._set_return_timestamps(
            return_timestamps=False, is_shortform=True, generation_config=generation_config
        )
        self.model._set_language_and_task(
            language=self.config.langid, task="transcribe", is_multilingual=True, generation_config=generation_config
        )
        self.model._set_num_frames(
            return_token_timestamps=True, generation_config=generation_config, kwargs=configs
        )
        self.model._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=None,
            compression_ratio_threshold=None,
            no_speech_threshold=None,
            condition_on_prev_tokens=False,
        )
        self.model._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type='first-segment',
        )
        generation_config.forced_decoder_ids = None # this argument is deprecated

        return generation_config
    
    def longest_common_substring(self, s1: list, s2: list) -> list:
        """
        返回 s1 和 s2 的最长公共子串。如果有多个等长结果，返回其中任意一个。
        """
        m, n = len(s1), len(s2)
        # dp[i][j] 表示以 s1[i-1] 和 s2[j-1] 结尾的最长公共子串长度
        dp = [[0] * (n+1) for _ in range(m+1)]
        max_len = 0       # 记录最长长度
        end_pos = 0       # 记录最长子串在 s1 中结束的位置

        # 遍历所有 i, j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    # 如果末字符相等，则在对角线基础上 +1
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        end_pos = i
                # 如果不相等，则 dp[i][j] 保持为 0（自动初始化）
        
        # 根据 end_pos 和 max_len 从 s1 中截取子串
        return s1[end_pos - max_len : end_pos], end_pos, max_len
    
    def longest_common_prefix(self, s1: str, s2: str) -> str:
        """返回 s1 和 s2 的最长公共前缀。"""
        # 找到最短字符串长度，避免越界
        min_len = min(len(s1), len(s2))
        # 从头逐字符比较
        i = 0
        while i < min_len and s1[i] == s2[i]:
            i += 1
        # 前 i 个字符都是相同的
        return s1[:i], i
    
    def seek_timestamp(self, offset=3):
        """"
        cut part of cached audio based on token timestamps
        params:
            offset: retain this number of second of audio
        """

        audio = torch.cat(self.chunk_cache, dim=0)
        tokens = torch.cat([self.init_tokens, *self.trust_tokens], dim=1).squeeze(0)
        tts = self.token_timestamps[:self.curr_token_length].tolist() # list is more friednly for while loop
        index = -1
        while True:
            if tts[index] < self.cur_segment_length - offset:
                break
            index -= 1
        cut_point = int(tts[index] * SAMPLE_RATE)
        return audio[cut_point:], tokens[index:]
    
    def transcribe(self, seg, is_last=False):
        # refresh cache when cache overflow
        if len(self.chunk_cache) > self.cache_length and not is_last:
            print(f"***REFRESH***")
            self.refresh_segment()

        # add chunk to cache
        self.chunk_cache.append(seg)

        # extract features use processor
        inputs = self.processor(
            torch.cat(self.chunk_cache, dim=0),
            padding="max_length",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
            return_token_timestamps=True,
        ).to(self.device)

        # set configs
        generation_config = self.set_infer_configs(inputs)

        # prepare logits processors
        logits_processor = self.model._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=None,
            begin_index=4,  # begin index is index of first generated decoder token
            num_beams=1,
            device=self.device,
        )

        # prepare decoder input ids
        decoder_input_ids = torch.cat([self.init_tokens, *self.trust_tokens], dim=1)

        # generate
        # s = time.time()
        (
            seek_sequences,
            seek_outputs,
            should_skip,
            do_condition_on_prev_tokens,
            model_output_type,
        ) = self.model.generate_with_fallback(
            segment_input=inputs["input_features"].to(self.infer_dtype),
            decoder_input_ids=decoder_input_ids,
            cur_bsz=1,
            generation_config=generation_config,
            return_token_timestamps=True,
            batch_idx_map=[0],
            seek=torch.tensor([0]),
            num_segment_frames=3000,
            max_frames=torch.tensor([3000]),
            temperatures=[None],
            logits_processor=logits_processor,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            do_condition_on_prev_tokens=[None],
            is_shortform=True, # always set to true
            batch_size=1,
            attention_mask=inputs["attention_mask"].to(self.infer_dtype),
            kwargs={},
        )
        # e = time.time()
        # print(f"!!!!!!!!!!infering time {e - s:.3f}s")
        _new_tokens = seek_sequences[0][self.curr_token_length:]

        # record intact token timestamps
        self.token_timestamps = seek_outputs[0]["token_timestamps"][:-1] # just save the start timestamps of every token
        self.cur_segment_length = inputs["num_frames"].item() / 100

        # last chunk, reset status and return
        if is_last:
            self.curr_token_length = 4
            self.trust_tokens = []
            self.first_chunk = True
            self.chunk_cache = []
            text = self.processor.tokenizer.decode(_new_tokens, skip_special_tokens=True)
            return text

        # first chunk, just record
        if self.first_chunk:
            self.last_tokens = _new_tokens
            self.first_chunk = False
            return ''

        # extract trusted tokens based on longest common prefix
        lcp, length = self.longest_common_prefix(self.last_tokens.int().tolist(), _new_tokens.int().tolist())
        # a = self.processor.tokenizer.decode(_new_tokens, skip_special_tokens=True)
        # b = self.processor.tokenizer.decode(self.last_tokens, skip_special_tokens=True)
        # c = self.processor.tokenizer.decode(torch.tensor(lcp), skip_special_tokens=True)
        # print(f"last_tokens {b}")
        # print(f"_new_tokens {a}")
        # print(f"lcp {c}, length {length}\n")

        if len(lcp) > 0:
            # keep last_tokens as the tail of _new_tokens
            self.last_tokens = _new_tokens[length:]
            _new_tokens = _new_tokens.new_tensor(lcp)
            self.empty_out = 0
        else:
            # output immediately
            if self.empty_out >= self.tolerance:
                self.last_tokens = _new_tokens.new_tensor([])
                self.empty_out = 0
            else:
                self.last_tokens = _new_tokens
                self.empty_out += 1
                return ''
        
        # renew token states
        self.curr_token_length += _new_tokens.shape[0]
        self.trust_tokens.append(_new_tokens.unsqueeze(0))

        # decode tokens
        text = self.processor.tokenizer.decode(_new_tokens, skip_special_tokens=True)

        # print("seek_sequences", seek_sequences)
        # print(f"text {text}")
        return text


if __name__ == "__main__":
    # hparams
    audio_path = 'dataset/1_16k.wav'
    config = StreamingConfig(
            chunk_length = 1.0,
            cache_length = 5,
            langid = 'en',
            model_path = 'models/whisper-large-v3',
            tolerance = 2,
            )
    transcriber = StreamingWhisperTransformers(config)

    # segment audio into chunks
    segmented_audio = SegmentWrapper(audio_path=audio_path, segment_length=config.chunk_length)

    out_t = []
    for seg_id, (seg, is_last) in enumerate(segmented_audio):
        # print(seg_id, '\n')
        text = transcriber.transcribe(seg, is_last)
        out_t.append(text)
        print(f"{''.join(out_t)}")