CREATE DATABASE IF NOT EXISTS multimodal_ai;
USE multimodal_ai;

CREATE TABLE IF NOT EXISTS inference_results (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_name VARCHAR(255) NOT NULL,
    query_text TEXT NOT NULL,
    image_label VARCHAR(255) NOT NULL,
    image_confidence FLOAT NOT NULL,
    face_count INT NOT NULL,
    edge_density FLOAT NOT NULL,
    token_count INT NOT NULL,
    sentiment_label VARCHAR(32) NOT NULL,
    sentiment_score FLOAT NOT NULL,
    combined_summary TEXT NOT NULL
);

SET @db_name = DATABASE();

SET @idx_exists = (
    SELECT COUNT(1)
    FROM information_schema.statistics
    WHERE table_schema = @db_name
      AND table_name = 'inference_results'
      AND index_name = 'idx_inference_created_at'
);
SET @sql = IF(
    @idx_exists = 0,
    'CREATE INDEX idx_inference_created_at ON inference_results (created_at)',
    'SELECT "idx_inference_created_at already exists"'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @idx_exists = (
    SELECT COUNT(1)
    FROM information_schema.statistics
    WHERE table_schema = @db_name
      AND table_name = 'inference_results'
      AND index_name = 'idx_inference_image_label'
);
SET @sql = IF(
    @idx_exists = 0,
    'CREATE INDEX idx_inference_image_label ON inference_results (image_label)',
    'SELECT "idx_inference_image_label already exists"'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @idx_exists = (
    SELECT COUNT(1)
    FROM information_schema.statistics
    WHERE table_schema = @db_name
      AND table_name = 'inference_results'
      AND index_name = 'idx_inference_sentiment_label'
);
SET @sql = IF(
    @idx_exists = 0,
    'CREATE INDEX idx_inference_sentiment_label ON inference_results (sentiment_label)',
    'SELECT "idx_inference_sentiment_label already exists"'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @idx_exists = (
    SELECT COUNT(1)
    FROM information_schema.statistics
    WHERE table_schema = @db_name
      AND table_name = 'inference_results'
      AND index_name = 'idx_inference_query_text'
);
SET @sql = IF(
    @idx_exists = 0,
    'CREATE FULLTEXT INDEX idx_inference_query_text ON inference_results (query_text)',
    'SELECT "idx_inference_query_text already exists"'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
