import asyncio
import os
import logging
import platform

from functools import wraps
from typing import Type

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, MetaData, text
from app.model.BackgroundTask import BackgroundTask, TaskStatus, TaskType
import time
import aiosqlite
from app.model.Chat import ChatMessage, ChatConversation
from sqlalchemy.exc import SQLAlchemyError

from app.model.knowledge import Knowledge
from app.model.note import Note
from sqlalchemy.ext.asyncio import AsyncEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

user_home = os.path.expanduser("~")

# DATABASE_PATH = os.path.join(user_home, 'Library/com.signerlabs.klee/db/klee.sqlite')
DATABASE_PATH = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/db/klee.sqlite"

if os.name == "posix":
    if 'darwin' in platform.system().lower():
        DATABASE_PATH = os.path.join(user_home, 'Library/Application Support/com.signerlabs.klee/db/klee.sqlite')

DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
Base = declarative_base()

async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

def db_transaction(func):
    """
    A decorator to wrap a function that performs database operations in a transaction.
    The function should accept a "session" keyword argument, which will be an AsyncSession object.
    If the "session" argument is not provided, a new AsyncSession object will be created and used.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):

        if kwargs.get("session") is not None:
            return await func(*args, **kwargs)

        async with async_session() as session:  # type: ignore
            async with session.begin():
                try:
                    kwargs.update({"session": session})

                    _result = await func(*args, **kwargs)
                    await session.commit()

                    return _result
                except Exception as e:
                    await session.rollback()
                    raise e

    return wrapper


async def _get_first(_id: str, Clazz: Type, session):
    """
    simple get first
    :param session: AsyncSession
    """
    result = await session.execute(select(Clazz).filter_by(id=_id))
    record = result.scalar_one_or_none()

    if record is None:
        raise ValueError(f"{Clazz.__name__} with id {_id} not found")

    return record


async def check_database_exists():
    """
    Check if the database file exists.
    """
    logger.info(f"Database file exists: {DATABASE_PATH}")
    if os.path.exists(DATABASE_PATH):
        print(f"Database file exists: {DATABASE_PATH}")
        return True
    else:
        print(f"Database file not found: {DATABASE_PATH}")
        return False


async def connect_to_database():
    """
    Connect to the SQLite database
    """
    while True:
        try:
            if await check_database_exists():
                async with aiosqlite.connect(DATABASE_PATH) as db:
                    logger.info(f"Successfully connected to database: {DATABASE_PATH}")
                    return db
            else:
                logger.info(f"Database file not found: {DATABASE_PATH}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {DATABASE_PATH} | {e}")

        await asyncio.sleep(4)


@db_transaction
async def create_task(task_type: TaskType, payload: str, session: AsyncSession = None):
    task = BackgroundTask(
        type=task_type,
        status=TaskStatus.CREATED,
        payload=payload,
        progress=0,
        create_at=int(time.time()),
        update_at=int(time.time())
    )
    session.add(task)

    return task


@db_transaction
async def update_task_status(task_id: str, status: TaskStatus, session: AsyncSession = None):
    task = await _get_first(task_id, BackgroundTask, session)

    # 更新任务状态
    task.status = status
    if status == TaskStatus.DONE:
        task.progress = 1
    task.update_at = int(time.time())

    return task


async def update_task_progress(task_id: str, progress: float):
    async with async_session() as session:
        async with session.begin():
            result = await session.execute(select(BackgroundTask).filter_by(id=task_id))
            task = result.scalar_one_or_none()
            if task is None:
                raise ValueError(f"Task with id {task_id} not found")

            task.progress = progress
            task.update_at = int(time.time())

            await session.commit()
            return task


async def add_task_progress(task_id: str, progress_will_add: float, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            async with async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        select(BackgroundTask).filter_by(id=task_id).with_for_update()
                    )
                    task = result.scalar_one_or_none()
                    if task is None:
                        raise ValueError(f"Task with id {task_id} not found")

                    new_progress = min(task.progress + progress_will_add, 1)
                    current_time = int(time.time())

                    result = await session.execute(
                        update(BackgroundTask)
                        .where(BackgroundTask.id == task_id, BackgroundTask.progress == task.progress)
                        .values(progress=new_progress, update_at=current_time)
                    )

                    if result.rowcount == 0:
                        await session.rollback()
                        continue

                    await session.commit()
                    return task

        except SQLAlchemyError as e:
            if attempt == max_retries - 1:
                raise e
            await session.rollback()
            continue

    raise Exception(f"Failed to update task progress after {max_retries} attempts")


async def main():
    db = await connect_to_database()
    async with db.execute('SELECT 1') as cursor:
        async for row in cursor:
            logger.info(f"Database version: {row}")


async def fetch_conversation_history_messages(conversation_id: str):
    async with async_session() as session:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.create_at.desc())
            .limit(10)
        )
        result = await session.execute(stmt)
        messages = result.scalars().all()
        return messages


async def fetch_conversation_info(conversation_id: str) -> ChatConversation:
    async with async_session() as session:
        stmt = (
            select(ChatConversation)
            .where(ChatConversation.id == conversation_id)
        )
        result = await session.execute(stmt)
        conversation = result.scalars().first()
        return conversation


async def fetch_note(id: str):
    async with async_session() as session:
        stmt = (
            select(Note)
            .where(Note.id == id)
        )
        result = await session.execute(stmt)
        note = result.scalars().first()
        return note


async def get_async_session():
    """
    Get a new async session
    """
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db(engine: AsyncEngine):
    db_exists = os.path.exists(DATABASE_PATH)

    if not db_exists:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    else:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # 只有在数据库已经存在的情况下才更新表结构
    if db_exists:
        await update_table_columns(Note.__tablename__)
        await update_table_columns(Knowledge.__tablename__)


async def update_table_columns(table_name: str):
    # 创建元数据对象并反射现有表结构
    metadata = MetaData()

    # 使用 run_sync 将异步引擎转换为同步引擎
    async with engine.begin() as conn:
        await conn.run_sync(metadata.reflect)

    # 检查表是否存在
    if table_name in metadata.tables:
        # 获取表对象
        table = metadata.tables[table_name]

        # 检查列是否已经存在
        if 'local_mode' not in table.columns:
            # 使用 ALTER TABLE 添加新列
            async with engine.begin() as conn:
                # 使用 text() 包装 SQL 语句
                stmt = text(f'ALTER TABLE {table_name} ADD COLUMN local_mode INTEGER DEFAULT 1')
                await conn.execute(stmt)

            logger.info(f"Column 'local_mode' added to table '{table_name}'.")
        else:
            logger.info(f"Column 'local_mode' already exists in table '{table_name}'.")


