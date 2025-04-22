from confluent_kafka import Producer
import csv
import json
from apify_client import ApifyClient
import os
import sys
import time
import threading

def read_config():
    return {'bootstrap.servers': 'localhost:9092'}  # Update with your Kafka server

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()}")

def store_url_in_dashboard(url):
    # Check if dashboard.py exists
    if os.path.exists('dashboard.py'):
        # Read the file content
        with open('dashboard.py', 'r') as file:
            content = file.read()
        
        # Check if URL variable already exists
        if 'URL =' in content or 'URL=' in content:
            # Replace the existing URL value
            import re
            updated_content = re.sub(r'URL\s*=\s*["\'].*?["\']', f'URL = "{url}"', content)
        else:
            # If URL variable does not exist, create a new one
            updated_content = content + f'\nURL = "{url}"\n'
    else:
        # If the file does not exist, create it with the URL
        updated_content = f'URL = "{url}"\n'
    
    # Write the updated content back to the file
    with open('dashboard.py', 'w') as file:
        file.write(updated_content)

def arc_spinner(stop_event, message, complete_message="Done!"):
    """Display an arc spinner animation"""
    spinner = ['◜', '◠', '◝', '◞', '◡', '◟']  # Arc spinner
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{spinner[i]} {message}")
        sys.stdout.flush()
        i = (i + 1) % len(spinner)
        time.sleep(0.1)
    
    sys.stdout.write(f"\r✓ {complete_message}\n")
    sys.stdout.flush()

def typing_effect(message, speed=0.03):
    """Display text with a typing effect"""
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    sys.stdout.write("\n")
    sys.stdout.flush()

def scrape_reviews(url):
    # Set up the loading animation
    stop_animation = threading.Event()
    animation_thread = threading.Thread(
        target=arc_spinner, 
        args=(stop_animation, f"Scraping reviews from Flipkart", "Reviews scraped successfully")
    )
    animation_thread.daemon = True
    animation_thread.start()
    
    try:
        # Show URL with typing effect
        typing_effect(f"Processing URL: {url}", speed=0.01)
        
        # Initialize the ApifyClient with your API token
        client = ApifyClient(" YOUR API TOKEN HERE")
        
        # Use a Flipkart scraper Actor
        # Using the Flipkart review extractor actor ID - this actor is specifically for Flipkart
        run_input = {
            "start_urls": [{"url": url}],
            "maxItems": 200,
            "proxyConfiguration": {
                "useApifyProxy": True
            }
        }
        
        # Run the actor for Flipkart review extraction
        run = client.actor("COcmxYbB46nexspPD").call(run_input=run_input)
        
        # Fetch Actor results from the run's dataset
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        
        # Stop the animation
        stop_animation.set()
        animation_thread.join()
        
        # Check if there are any items to export
        if items:
            # Define the CSV file name
            csv_file_name = "flipkart_results.csv"
            
            # Start a new animation for CSV export
            stop_animation = threading.Event()
            animation_thread = threading.Thread(
                target=arc_spinner, 
                args=(stop_animation, "Exporting data to CSV", "CSV export complete!")
            )
            animation_thread.daemon = True
            animation_thread.start()
            
            # Open the CSV file for writing
            with open(csv_file_name, mode='w', newline='', encoding='utf-8') as csv_file:
                # Get all possible keys from all items
                all_keys = set()
                for item in items:
                    all_keys.update(item.keys())
                
                # Create a CSV writer object
                writer = csv.DictWriter(csv_file, fieldnames=list(all_keys))
                
                # Write the header
                writer.writeheader()
                
                # Write the data rows
                for item in items:
                    # Fill in missing keys with empty strings
                    row_data = {key: item.get(key, "") for key in all_keys}
                    writer.writerow(row_data)
            
            # Stop the animation
            stop_animation.set()
            animation_thread.join()
            
            print(f"✓ Found {len(items)} reviews and exported to {csv_file_name}")
            return csv_file_name
        else:
            print("⚠ No items found to export.")
            return None
            
    except Exception as e:
        # Stop the animation in case of error
        stop_animation.set()
        animation_thread.join()
        print(f"❌ Error during scraping: {str(e)}")
        return None

def produce_to_kafka(csv_file_name):
    # Set up the loading animation
    stop_animation = threading.Event()
    animation_thread = threading.Thread(
        target=arc_spinner, 
        args=(stop_animation, "Sending data to Kafka", "All messages sent to Kafka!")
    )
    animation_thread.daemon = True
    animation_thread.start()
    
    try:
        config = read_config()
        producer = Producer(config)
        topic = "customer_reviews"
        
        with open(csv_file_name, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert row to JSON and send to Kafka
                producer.produce(
                    topic,
                    key=str(row.get('id', '')),
                    value=json.dumps(row),
                    callback=delivery_report
                )
                producer.poll(0)
                time.sleep(0.01)
                
        producer.flush()
        
        # Stop the animation
        stop_animation.set()
        animation_thread.join()
        
    except Exception as e:
        # Stop the animation in case of error
        stop_animation.set()
        animation_thread.join()
        print(f"❌ Error while producing to Kafka: {str(e)}")

def display_welcome():
    """Display a welcome message with animation"""
    welcome_text = """
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║      FLIPKART REVIEW SCRAPER & ANALYZER       ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
    """
    
    for line in welcome_text.split('\n'):
        print(line)
        time.sleep(0.1)
    
    print("\nThis tool scrapes Flipkart product reviews and sends them to Kafka for analysis.\n")

def main():
    try:
        display_welcome()
        
        url = input("🔗 Enter Flipkart product URL: ")
        print()
        
        # Store the URL in dashboard.py
        stop_animation = threading.Event()
        animation_thread = threading.Thread(
            target=arc_spinner, 
            args=(stop_animation, "Storing URL in dashboard", "URL stored successfully")
        )
        animation_thread.daemon = True
        animation_thread.start()
        
        store_url_in_dashboard(url)
        
        stop_animation.set()
        animation_thread.join()
        
        # Continue with the original process
        csv_file_name = scrape_reviews(url)
        if csv_file_name:
            produce_to_kafka(csv_file_name)
            
        print("\n✨ Process completed successfully! ✨\n")
            
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
