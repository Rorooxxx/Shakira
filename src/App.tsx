import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from "@google/genai";
import { Mic, MicOff, Volume2, VolumeX, Loader2, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// Constants for Audio
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef(false);

  const systemInstruction = `You are named "Shakira", a voice agent for Romain. 
You speak English and French with a British accent. 
You are helpful, witty, and knowledgeable about Romain.
Your first words when the session starts MUST be: "hello, I'm Shakira Lolé Lolé Lolé, what would you like to know about Romain"
Keep your responses concise and conversational.`;

  const stopAudio = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsSpeaking(false);
    isPlayingRef.current = false;
    audioQueueRef.current = [];
  }, []);

  const playNextChunk = useCallback(() => {
    if (audioQueueRef.current.length === 0 || !audioContextRef.current) {
      isPlayingRef.current = false;
      setIsSpeaking(false);
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);
    const chunk = audioQueueRef.current.shift()!;
    const float32Data = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) {
      float32Data[i] = chunk[i] / 32768.0;
    }

    const buffer = audioContextRef.current.createBuffer(1, float32Data.length, OUTPUT_SAMPLE_RATE);
    buffer.getChannelData(0).set(float32Data);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    source.onended = playNextChunk;
    source.start();
  }, []);

  const connect = async () => {
    if (isConnected) {
      sessionRef.current?.close();
      stopAudio();
      setIsConnected(false);
      return;
    }

    setIsConnecting(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      // Setup Audio Context
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: OUTPUT_SAMPLE_RATE
      });
      
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      // Setup Microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const source = audioContextRef.current.createMediaStreamSource(stream);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (isMuted || !sessionRef.current || !audioContextRef.current) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        const inputSampleRate = audioContextRef.current.sampleRate;
        const outputSampleRate = 16000;
        
        // Simple linear resampling
        const ratio = inputSampleRate / outputSampleRate;
        const newLength = Math.floor(inputData.length / ratio);
        const resampledData = new Int16Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
          const index = Math.floor(i * ratio);
          const nextIndex = Math.min(index + 1, inputData.length - 1);
          const weight = (i * ratio) - index;
          
          // Linear interpolation
          const interpolated = inputData[index] * (1 - weight) + inputData[nextIndex] * weight;
          resampledData[i] = Math.max(-1, Math.min(1, interpolated)) * 32767;
        }
        
        // Efficient Base64 conversion
        let binary = '';
        const bytes = new Uint8Array(resampledData.buffer);
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const base64Data = btoa(binary);

        sessionRef.current.sendRealtimeInput({
          audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
        });
      };

      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      // Connect to Gemini Live
      const session = await ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-12-2025",
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          systemInstruction,
        },
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            setIsConnecting(false);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.modelTurn?.parts) {
              for (const part of message.serverContent.modelTurn.parts) {
                if (part.inlineData?.data) {
                  const binaryString = atob(part.inlineData.data);
                  const bytes = new Uint8Array(binaryString.length);
                  for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                  }
                  const int16Data = new Int16Array(bytes.buffer);
                  audioQueueRef.current.push(int16Data);
                  if (!isPlayingRef.current) {
                    playNextChunk();
                  }
                }
              }
            }
            
            if (message.serverContent?.interrupted) {
              audioQueueRef.current = [];
              setIsSpeaking(false);
            }
          },
          onclose: () => {
            setIsConnected(false);
            stopAudio();
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setIsConnected(false);
            stopAudio();
          }
        }
      });

      sessionRef.current = session;

    } catch (err) {
      console.error("Connection failed:", err);
      setIsConnecting(false);
      stopAudio();
    }
  };

  useEffect(() => {
    return () => {
      stopAudio();
    };
  }, [stopAudio]);

  return (
    <div className="min-h-screen bg-[#0a0502] text-[#e0d8d0] font-sans selection:bg-[#ff4e00]/30 overflow-hidden flex flex-col items-center justify-center p-6 relative">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[60%] h-[60%] bg-[#3a1510] rounded-full blur-[120px] opacity-40 animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] bg-[#ff4e00] rounded-full blur-[150px] opacity-20" />
      </div>

      <main className="relative z-10 w-full max-w-2xl flex flex-col items-center gap-12">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-2"
        >
          <h1 className="text-5xl font-serif italic tracking-tight text-white">Shakira</h1>
          <p className="text-sm uppercase tracking-[0.3em] text-[#ff4e00] font-medium opacity-80">Voice Agent for Romain</p>
        </motion.div>

        <div className="relative w-64 h-64 flex items-center justify-center">
          <AnimatePresence mode="wait">
            {isConnected ? (
              <motion.div
                key="active"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                className="relative"
              >
                {isSpeaking && (
                  <>
                    <motion.div 
                      animate={{ scale: [1, 1.5, 1], opacity: [0.3, 0.1, 0.3] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="absolute inset-0 border border-[#ff4e00] rounded-full"
                    />
                    <motion.div 
                      animate={{ scale: [1, 2, 1], opacity: [0.2, 0, 0.2] }}
                      transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
                      className="absolute inset-0 border border-[#ff4e00] rounded-full"
                    />
                  </>
                )}
                
                <div className="w-48 h-48 rounded-full bg-gradient-to-br from-[#1a0a05] to-[#0a0502] border border-[#ff4e00]/30 flex items-center justify-center shadow-[0_0_50px_rgba(255,78,0,0.1)]">
                  <Sparkles className={`w-12 h-12 ${isSpeaking ? 'text-[#ff4e00] animate-pulse' : 'text-[#e0d8d0]/40'}`} />
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="inactive"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                className="w-48 h-48 rounded-full border border-[#e0d8d0]/10 flex items-center justify-center"
              >
                <div className="w-40 h-40 rounded-full border border-[#e0d8d0]/5 flex items-center justify-center">
                  <VolumeX className="w-10 h-10 text-[#e0d8d0]/20" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="flex flex-col items-center gap-8 w-full">
          <div className="flex items-center gap-6">
            <button
              onClick={() => setIsMuted(!isMuted)}
              disabled={!isConnected}
              className={`p-4 rounded-full border transition-all duration-300 ${
                isMuted 
                  ? 'bg-[#ff4e00]/10 border-[#ff4e00] text-[#ff4e00]' 
                  : 'bg-white/5 border-white/10 text-white hover:bg-white/10 disabled:opacity-30'
              }`}
            >
              {isMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
            </button>

            <button
              onClick={connect}
              disabled={isConnecting}
              className={`px-10 py-4 rounded-full font-medium tracking-wide transition-all duration-500 flex items-center gap-3 ${
                isConnected
                  ? 'bg-transparent border border-white/20 text-white hover:bg-white/5'
                  : 'bg-white text-black hover:bg-[#ff4e00] hover:text-white shadow-[0_0_30px_rgba(255,255,255,0.1)]'
              }`}
            >
              {isConnecting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Connecting...
                </>
              ) : isConnected ? (
                'End Session'
              ) : (
                'Start Conversation'
              )}
            </button>

            <button
              className="p-4 rounded-full bg-white/5 border border-white/10 text-white hover:bg-white/10 transition-all"
            >
              <Volume2 className="w-6 h-6" />
            </button>
          </div>

          <div className="h-12 flex items-center justify-center text-center px-4">
            <AnimatePresence mode="wait">
              {!isConnected && !isConnecting && (
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.6 }}
                  exit={{ opacity: 0 }}
                  className="text-sm italic font-serif"
                >
                  "hello, I'm Shakira Lolé Lolé Lolé..."
                </motion.p>
              )}
              {isConnecting && (
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-sm uppercase tracking-widest text-[#ff4e00]"
                >
                  Establishing Connection
                </motion.p>
              )}
              {isConnected && isSpeaking && (
                <motion.p 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="text-lg font-serif italic text-white/90"
                >
                  Shakira is speaking...
                </motion.p>
              )}
              {isConnected && !isSpeaking && !isMuted && (
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.5 }}
                  className="text-sm uppercase tracking-widest"
                >
                  Listening
                </motion.p>
              )}
              {isConnected && isMuted && (
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-sm uppercase tracking-widest text-[#ff4e00]"
                >
                  Microphone Muted
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      <footer className="absolute bottom-8 left-0 right-0 flex justify-center opacity-30">
        <div className="flex items-center gap-4 text-[10px] uppercase tracking-[0.2em]">
          <span>English</span>
          <div className="w-1 h-1 rounded-full bg-white" />
          <span>Français</span>
          <div className="w-1 h-1 rounded-full bg-white" />
          <span>British Accent</span>
        </div>
      </footer>
    </div>
  );
}
