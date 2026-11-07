"""PhoWhisper Vietnamese speech recognition ROS 2 node for tb4_voice_nav."""

from __future__ import annotations

import threading
from typing import Callable, Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

from tb4_voice_nav.pulse_recorder import load_wav, record_pulse_audio
from tb4_voice_nav.voice_navigator_node import ensure_fastdds_udp_env


class PhoWhisperTranscriber:
    """Wrapper around HuggingFace Automatic Speech Recognition pipeline for VinAI PhoWhisper."""

    def __init__(
        self,
        model_name: str = 'vinai/PhoWhisper-small',
        device: int | str | None = None,
        pipeline_fn: Callable | None = None,
    ) -> None:
        self.model_name = model_name
        if pipeline_fn is not None:
            self._pipe = pipeline_fn
            return

        import PIL.Image
        if not hasattr(PIL.Image, 'Resampling'):
            PIL.Image.Resampling = PIL.Image

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif isinstance(device, int):
            self._device = f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu'
        else:
            self._device = str(device)

        self._pipe = None
        self._torch = torch
        self._processor = WhisperProcessor.from_pretrained(model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe 16kHz float32 mono audio into Vietnamese text."""
        audio_arr = np.asarray(audio_f32, dtype=np.float32)
        if self._pipe is not None:
            payload = {'raw': audio_arr, 'sampling_rate': int(sample_rate)}
            try:
                out = self._pipe(
                    payload,
                    generate_kwargs={'language': 'vi', 'task': 'transcribe'},
                )
            except TypeError:
                out = self._pipe(payload)
            if isinstance(out, dict):
                return str(out.get('text', '')).strip()
            return str(out).strip()

        inputs = self._processor(
            audio_arr,
            sampling_rate=int(sample_rate),
            return_tensors='pt',
        )
        input_features = inputs.input_features.to(self._device)
        with self._torch.no_grad():
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language='vi', task='transcribe'
            )
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )
        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return str(transcription[0] if transcription else '').strip()


class VoiceCommandNode(Node):
    """Capture microphone audio via PulseAudio, transcribe with PhoWhisper, publish text."""

    def __init__(self, transcriber: PhoWhisperTranscriber | None = None) -> None:
        super().__init__('voice_command_node')
        self.declare_parameter('model_name', 'vinai/PhoWhisper-small')
        self.declare_parameter('record_seconds', 4.5)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('interactive_prompt', True)
        self.declare_parameter('wav_file', '')

        self.model_name = str(self.get_parameter('model_name').value)
        self.record_seconds = float(self.get_parameter('record_seconds').value)
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.interactive_prompt = bool(self.get_parameter('interactive_prompt').value)
        self.wav_file = str(self.get_parameter('wav_file').value).strip()

        self.cmd_pub = self.create_publisher(String, '/voice_command_text', 10)
        self.trigger_sub = self.create_subscription(
            Empty,
            '/voice_trigger',
            self._on_trigger,
            10,
        )

        self._transcriber = transcriber
        self._busy_lock = threading.Lock()

        self.get_logger().info(
            f'Initializing PhoWhisper ASR model: {self.model_name} ...'
        )
        if self._transcriber is None:
            self._transcriber = PhoWhisperTranscriber(model_name=self.model_name)
        self.get_logger().info(
            'PhoWhisper ready! Publish to /voice_trigger or press ENTER to speak.'
        )

        if self.wav_file:
            self.transcribe_wav_file(self.wav_file)

        if self.interactive_prompt:
            self._prompt_thread = threading.Thread(
                target=self._interactive_loop, daemon=True
            )
            self._prompt_thread.start()

    def _on_trigger(self, _msg: Empty) -> None:
        threading.Thread(target=self.capture_and_publish, daemon=True).start()

    def transcribe_wav_file(self, wav_path: str) -> str:
        audio, sr = load_wav(wav_path)
        text = self._transcriber.transcribe(audio, sample_rate=sr)
        self.get_logger().info(f"[PhoWhisper WAV] '{text}'")
        if text:
            msg = String()
            msg.data = text
            self.cmd_pub.publish(msg)
        return text

    def capture_and_publish(self) -> str:
        if not self._busy_lock.acquire(blocking=False):
            self.get_logger().warn('Microphone recording already in progress.')
            return ''
        try:
            self.get_logger().info(
                f'>>> ĐANG THU ÂM ({self.record_seconds:.1f}s)... Hãy nói câu lệnh điều hướng!'
            )
            audio = record_pulse_audio(
                duration_sec=self.record_seconds,
                sample_rate=self.sample_rate,
            )
            rms = float(np.sqrt(np.mean(audio**2)))
            self.get_logger().info(
                f'Thu âm hoàn tất (RMS={rms:.4f}). Đang nhận dạng bằng {self.model_name}...'
            )
            text = self._transcriber.transcribe(audio, sample_rate=self.sample_rate)
            self.get_logger().info(f">>> KẾT QUẢ PHOWHISPER: '{text}'")
            if text:
                msg = String()
                msg.data = text
                self.cmd_pub.publish(msg)
            return text
        except Exception as exc:
            self.get_logger().error(f'Lỗi thu âm/nhận dạng: {exc}')
            return ''
        finally:
            self._busy_lock.release()

    def _interactive_loop(self) -> None:
        while rclpy.ok():
            try:
                user_in = input(
                    '\n[PhoWhisper] Nhấn ENTER để nói vào mic (hoặc gõ trực tiếp câu lệnh, q để thoát): '
                )
            except EOFError:
                break
            cmd = user_in.strip()
            if cmd.lower() in ('q', 'quit', 'exit'):
                self.get_logger().info('Thoát chế độ nhập lệnh.')
                break
            if cmd == '':
                self.capture_and_publish()
            else:
                self.get_logger().info(f">>> LỆNH TRỰC TIẾP: '{cmd}'")
                msg = String()
                msg.data = cmd
                self.cmd_pub.publish(msg)


def main(args: Sequence[str] | None = None) -> None:
    ensure_fastdds_udp_env()
    rclpy.init(args=args)
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
