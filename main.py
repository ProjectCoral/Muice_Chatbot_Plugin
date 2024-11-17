import importlib
import json
import logging
import sys
import os
import asyncio

script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.append(script_dir)

from Muice import Muice

logger = logging.getLogger(__name__)

if not os.path.exists('./config/Muice_Chatbot_Plugin'):
    os.makedirs('./config/Muice_Chatbot_Plugin')

if not os.path.exists('./data/Muice_Chatbot_Plugin'):
    os.makedirs('./data/Muice_Chatbot_Plugin')

if not os.path.exists('./config/Muice_Chatbot_Plugin/configs.json'):
    logger.error("配置文件不存在，创建默认配置文件")
    config_template = json.load(open('./plugins/Muice_Chatbot_Plugin/configs.json', 'r', encoding='utf-8'))
    json.dump(config_template, open('./config/Muice_Chatbot_Plugin/configs.json', 'w', encoding='utf-8'), indent=4)
    logger.info("请修改配置文件")
    raise FileNotFoundError("请修改配置文件")

configs = json.load(open('./config/Muice_Chatbot_Plugin/configs.json', 'r', encoding='utf-8'))

# 模型配置
model_loader = configs["model_loader"]
model_name_or_path = configs["model_name_or_path"]
adapter_name_or_path = configs["adapter_name_or_path"]

# 模型加载
model_adapter = importlib.import_module(f"llm.{model_loader}")
model = model_adapter.llm(model_name_or_path, adapter_name_or_path)

# Faiss配置
enable_faiss = configs.get('enable_faiss', False)
if enable_faiss:
    from llm.faiss_memory import FAISSMemory
    import signal
    memory = FAISSMemory(model_path=configs["sentence_transformer_model_name_or_path"],db_path="./data/Muice_Chatbot_Plugin/memory/faiss_index.faiss",top_k=2)
    def handle_interrupt(faiss_memory: FAISSMemory):
        """处理中断信号"""
        logging.info("接收到中断信号，正在保存数据...")
        faiss_memory.save_all_data()
        sys.exit(0)
    signal.signal(signal.SIGINT, lambda sig, frame: handle_interrupt(memory))
else:
    memory = None

# OFA图像模型
enable_ofa_image = configs["enable_ofa_image"]
if enable_ofa_image:
    from utils.ofa_image_process import ImageCaptioningPipeline
    ofa_image_model_name_or_path = configs["ofa_image_model_name_or_path"]
    ImageCaptioningPipeline.load_model(ofa_image_model_name_or_path)
    from utils.image_database import ImageDatabase
    image_db = ImageDatabase(db_name='./data/Muice_Chatbot_Plugin/image_data/image_data.db')


muice_app = Muice(model,memory,configs['read_memory_from_file'], configs['known_topic_probability'],
                  configs['time_topic_probability'])
def register_plugin(register, config, perm_system):
    perm_system.register_perm("muice_chatbot", "muice_chatbot_plugin base permisson")
    perm_system.register_perm("muice_chatbot.reply.group", "muice_chatbot_plugin group reply permission")
    perm_system.register_perm("muice_chatbot.reply.private", "muice_chatbot_plugin private reply permission")
    perm_system.register_perm("muice_chatbot.commands", "muice_chatbot_plugin commands permission")
    chatbot_instance = Chatbot(register, config, perm_system)
    register.register_function("process_text", chatbot_instance.chat)
    register.register_function("process_image", chatbot_instance.image_chat)
    register.register_function("store_memory", chatbot_instance.store_memory)
    register.register_command("refresh", "刷新对话记录", chatbot_instance.refresh_memory, ["muice_chatbot", "muice_chatbot.commands"])
    register.register_command("clear", "清空对话记录", chatbot_instance.clear_memory, ["muice_chatbot", "muice_chatbot.commands"])
    register.register_command("undo", "撤销对话", chatbot_instance.undo_memory, ["muice_chatbot", "muice_chatbot.commands"])

class Chatbot:
    register = None
    config = None
    perm_system = None

    def __init__(self, register, config, perm_system):
        self.register = register
        self.config = config
        self.perm_system = perm_system

    async def chat(self, message):
        text = message['message']
        user_qq = message['sender_user_id']
        group_id = message['group_id']

        if group_id == -1:
            if not self.perm_system.check_perm(["muice_chatbot", "muice_chatbot.reply.private"], user_qq, group_id):
                return {"message": None, "sender_user_id": user_qq, "group_id": group_id}
        else:
            if not self.perm_system.check_perm(["muice_chatbot", "muice_chatbot.reply.group"], '-1', group_id):
                return {"message": None, "sender_user_id": user_qq, "group_id": group_id}
            
        response = muice_app.ask(text, user_qq, group_id)

        if configs['Reply_Wait']:
            await asyncio.sleep(len(response) * 0.5)

        response =  await self.search_image(response)

        return {"message": response, "sender_user_id": user_qq, "group_id": group_id}
    
    async def store_memory(self, message):
        text = message['message']
        reply = message['reply']
        user_qq = message['sender_user_id']
        group_id = message['group_id']

        muice_app.finish_ask(reply)

        return "saved"
    
    async def image_chat(self, message):
        image_url = message['image_url']
        user_qq = message['sender_user_id']
        group_id = message['group_id']

        if not enable_ofa_image:
            logger.warning("OFA图像模型未启用，无法进行图像对话")
            return {"message": None, "sender_user_id": user_qq, "group_id": group_id}

        if group_id == -1:
            if not self.perm_system.check_perm(["muice_chatbot", "muice_chatbot.reply.private"], user_qq, group_id):
                return {"message": None, "sender_user_id": user_qq, "group_id": group_id}
        else:
            if not self.perm_system.check_perm(["muice_chatbot", "muice_chatbot.reply.group"], '-1', group_id):
                return {"message": None, "sender_user_id": user_qq, "group_id": group_id}

        message = await ImageCaptioningPipeline().generate_caption(image_url)
        await image_db.insert_data(message, image_url)
        message = f"(收到图片描述：{message})"

        response = muice_app.ask(message, user_qq, group_id)

        if configs['Reply_Wait']:
            await asyncio.sleep(len(response) * 0.5)

        response =  await self.search_image(response)

        return {"message": response, "sender_user_id": user_qq, "group_id": group_id}
    
    async def search_image(self, reply):
        if enable_ofa_image:
            similar_image = await image_db.find_similar_content(reply)
            if similar_image is not None and similar_image[1] is not None:
                if similar_image[1] > 0.6:
                    logging.info(f"找到相似图片：{similar_image[0]},相似度为{similar_image[1]}")
                    try:
                        url = similar_image[0].replace('&', '&amp;')
                    except:
                        pass
                    reply_list = [f"[CQ:image,url={url}]"]
                    return reply_list
        return reply
    
    def refresh_memory(self, *args):
        return muice_app.refresh()
    
    def clear_memory(self, *args):
        try:
            for file in os.listdir('./data/Muice_Chatbot_Plugin/memory/'):
                if file.endswith('.json'):
                    os.remove(os.path.join('./data/Muice_Chatbot_Plugin/memory/', file))
            return "memory cleared"
        except:
            return "memory clear failed"
        
    def undo_memory(self, *args):
        try:
            muice_app.remove_last_chat_memory()
            muice_app.history = muice_app.get_recent_chat_memory()
            return "memory undone"
        except:
            return "memory undo failed"
        

