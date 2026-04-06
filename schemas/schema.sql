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

CREATE INDEX idx_inference_created_at ON inference_results (created_at);
CREATE INDEX idx_inference_image_label ON inference_results (image_label);
CREATE INDEX idx_inference_sentiment_label ON inference_results (sentiment_label);
CREATE FULLTEXT INDEX idx_inference_query_text ON inference_results (query_text);
