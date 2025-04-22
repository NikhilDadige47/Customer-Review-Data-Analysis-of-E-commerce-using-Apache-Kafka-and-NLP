from confluent_kafka import Consumer, KafkaError, KafkaException
from transformers import pipeline
import json
import pandas as pd
import time

# Initialize sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def read_config():
    return {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'review-group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,  # Manual commit for better control
        'session.timeout.ms': 6000,  # Increased timeout
        'max.poll.interval.ms': 600000  # Increased for ML processing
    }

def main():
    config = read_config()
    consumer = Consumer(config)
    
    # Create the topic if it doesn't exist (optional)
    topic = 'customer_reviews'
    consumer.subscribe([topic])
    
    enriched_data = []
    
    try:
        while True:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue  # Skip if no message is received
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition, not an error
                    continue
                else:
                    raise KafkaException(msg.error())
                
            try:
                # Parse message
                value = msg.value().decode('utf-8')  # Decode bytes to string
                review = json.loads(value)
                
                # Compute sentiment
                text = review.get('text', '')
                if text:
                    sentiment = sentiment_pipeline(text)[0]
                    review['sentiment_label'] = sentiment['label']
                    review['sentiment_score'] = sentiment['score']
                else:
                    review['sentiment_label'] = 'UNKNOWN'
                    review['sentiment_score'] = 0.0
                    
                enriched_data.append(review)
                
                # Acknowledge the processed message
                print(f"Processed review: {review['id']} ")
                
                # Manually commit offset
                consumer.commit(msg)
                
            except Exception as e:
                # Handle any processing errors silently
                print(f"Error processing message: {e}")
                continue
            
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        # Save enriched data to a new CSV
        if enriched_data:
            df = pd.DataFrame(enriched_data)
            df.to_csv('enriched_reviews.csv', index=False)
            print(f"Data saved to enriched_reviews.csv with {len(enriched_data)} reviews")
        else:
            print("No data processed")

if __name__ == "__main__":
    main()
