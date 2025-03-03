CREATE_TABLE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS chat_conversation (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        latestMessage TEXT NOT NULL,
        knowledgeIds TEXT NOT NULL,
        noteIds TEXT NOT NULL,
        fileIds TEXT NOT NULL,
        createAt REAL NOT NULL,
        updateAt REAL NOT NULL,
        systemPrompt TEXT NOT NULL,
        replyLanguage TEXT NOT NULL,
        llmId TEXT NOT NULL,
        knowledgeContent TEXT NOT NULL,
        isPin INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_message (
        id TEXT PRIMARY KEY,
        conversationId TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        status TEXT NOT NULL,
        createAt REAL NOT NULL,
        updateAt REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llama_chat_message (
        id TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        content TEXT NOT NULL,
        create_time REAL NOT NULL,
        modify_time REAL,
        status TEXT NOT NULL,
        error_message TEXT,
        create_at REAL,
        update_at REAL,
        delete_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llama_chat_conversation (
        id TEXT PRIMARY KEY,
        knowledge_ids TEXT NOT NULL,
        note_ids TEXT NOT NULL,
        title TEXT NOT NULL,
        create_time REAL NOT NULL,
        is_pin INTEGER NOT NULL DEFAULT 0,
        local_mode INTEGER NOT NULL DEFAULT 0,
        provider_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        language_id TEXT NOT NULL,
        system_prompt TEXT DEFAULT '',
        model_path TEXT DEFAULT '',
        model_name TEXT DEFAULT '',
        create_at REAL,
        update_at REAL,
        delete_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS base_config (
        id TEXT PRIMARY KEY,
        apiKey TEXT DEFAULT '',
        name TEXT DEFAULT '',
        description TEXT DEFAULT '',
        baseUrl TEXT DEFAULT '',
        models TEXT DEFAULT '',
        create_at REAL,
        update_at REAL,
        delete_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge (
        id TEXT PRIMARY KEY,
        timeStamp REAL NOT NULL,
        title TEXT NOT NULL,
        icon TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL,
        isPin INTEGER NOT NULL DEFAULT 0,
        folder_path TEXT NOT NULL DEFAULT '',
        embed_status TEXT NOT NULL DEFAULT 'embedding',
        parent_id TEXT DEFAULT '',
        create_at REAL,
        update_at REAL,
        delete_at REAL,
        local_mode INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS file (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        knowledgeId TEXT,
        format TEXT NOT NULL,
        size INTEGER NOT NULL,
        uploaded INTEGER NOT NULL,
        path TEXT NOT NULL,
        conversationId TEXT,
        os_mtime REAL NOT NULL DEFAULT 0,
        path_type TEXT NOT NULL DEFAULT 'RELATIVE',
        create_at REAL,
        update_at REAL,
        delete_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS background_task (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        status TEXT NOT NULL,
        payload TEXT,
        progress REAL,
        create_at INTEGER NOT NULL,
        update_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_message_rag_source (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        file_id TEXT,
        file_format TEXT,
        knowledge_id TEXT,
        keywords TEXT,
        page INTEGER,
        total_pages INTEGER,
        message_id TEXT,
        conversation_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS global_setting_table (
        id TEXT PRIMARY KEY,
        local_mode INTEGER NOT NULL DEFAULT 0,
        model_id TEXT,
        model_name TEXT,
        model_path TEXT,
        provider_id TEXT,
        create_at INTEGER NOT NULL,
        update_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS note (
        id TEXT PRIMARY KEY,
        folder_id TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        type TEXT NOT NULL,
        status TEXT NOT NULL,
        is_pin INTEGER NOT NULL,
        create_at REAL NOT NULL,
        update_at REAL NOT NULL,
        delete_at REAL NOT NULL DEFAULT 0,
        html_content TEXT NOT NULL DEFAULT '',
        embedded_content TEXT NOT NULL DEFAULT '',
        local_mode INTEGER NOT NULL DEFAULT 0
    )
    """
]