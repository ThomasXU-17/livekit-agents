import logging
import os
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomIO,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import ChatContext
from livekit.agents.tts import StreamAdapter
from livekit.plugins import openai, silero
from livekit.plugins.openfacebot import STT, TTS
from livekit.agents.tokenize import basic

from datetime import datetime
import json

logger = logging.getLogger("openfacebot-agent")

load_dotenv(".env.local")

# 定义你的 AI 角色
character = {
    "assistant": "你是一个友好的 AI 助手，名叫小助手。你可以用中文流畅地与用户交流，解答问题，提供帮助。请保持回答简洁、友好。",
}


class MyAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(
            instructions=instructions,
            chat_ctx=ChatContext.empty()
        )

    async def on_enter(self):
        self.session.generate_reply()


def prewarm(proc: JobProcess):
    # 使用 Silero VAD (内置，不需要额外安装)
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(f"connecting to room {ctx.room.name}")

    # AI 角色设定
    name = "assistant"
    instruction = character.get(name, character["assistant"])
    
    # 创建 Agent
    used_agent = MyAgent(instruction)
    
    # 使用 Qwen-Plus LLM (通过 DashScope)
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY not found in environment variables")
    
    used_llm = openai.LLM(
        model="qwen-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=dashscope_api_key,
    )

    # 创建 AgentSession
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=used_llm,
        stt=STT(
            model="paraformer-realtime-v2",
            language="zh",
            api_key=dashscope_api_key,
        ),
        tts=TTS(
            model="cosyvoice-v3-flash",
            voice="longanhuan",
            api_key=dashscope_api_key,
            response_format="pcm",
            sample_rate=16000,
        ),
    )

    # 保存对话记录
    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_dir}/transcript_{ctx.room.name}_{current_date}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(session.chat_ctx.to_dict(), f, indent=4, ensure_ascii=False)
            logger.info(f"Transcript saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")

    ctx.add_shutdown_callback(write_transcript)

    # 连接到房间
    await ctx.connect()
    
    # 启动 Agent Session
    await session.start(
        agent=used_agent,
        room=ctx.room,
    )
    
    logger.info(f"Agent started in room {ctx.room.name}")
    
    # 等待参与者加入
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # RPC 方法：重置对话
    @ctx.room.local_participant.register_rpc_method("new_conversation")
    async def new_conversation(data: rtc.RpcInvocationData):
        logger.info("Starting new conversation")
        session.chat_ctx.messages.clear()
        return json.dumps({"status": "ok"})


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))