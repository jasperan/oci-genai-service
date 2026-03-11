-- Initialize vector tables for oci-genai-service

CREATE TABLE IF NOT EXISTS documents (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    text CLOB,
    embedding VECTOR(1024, FLOAT64),
    metadata JSON,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP
);

CREATE VECTOR INDEX IF NOT EXISTS idx_documents_vec
ON documents (embedding)
ORGANIZATION NEIGHBOR PARTITIONS
DISTANCE COSINE
WITH TARGET ACCURACY 95;

CREATE TABLE IF NOT EXISTS conversations (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    session_id VARCHAR2(255) NOT NULL,
    role VARCHAR2(20) NOT NULL,
    content CLOB,
    embedding VECTOR(1024, FLOAT64),
    metadata JSON,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_session
ON conversations (session_id, created_at);

CREATE VECTOR INDEX IF NOT EXISTS idx_conversations_vec
ON conversations (embedding)
ORGANIZATION NEIGHBOR PARTITIONS
DISTANCE COSINE
WITH TARGET ACCURACY 95;

COMMIT;
