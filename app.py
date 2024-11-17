# Apply eventlet monkey-patching FIRST
import eventlet
eventlet.monkey_patch()

# Import other modules AFTER monkey-patching
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_socketio import SocketIO, emit
import threading
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from utils.predict_quadrant import QuadrantPredictor
from utils.sample_mood_model import MoodModel
import cv2
import time
import signal
import sys

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(16)
app.config['WTF_CSRF_ENABLED'] = False

# Initialize SocketIO with the app
socketio = SocketIO(app, cors_allowed_origins="*")

# Load or regenerate the mood detection model
model_path = os.path.join('models', 'mood_model.pkl')
def load_or_regenerate_model(regenerate=False):
    if regenerate or not os.path.exists(model_path):
        print("Training a new mood model...")
        mood_model_obj = MoodModel()
        mood_model_obj.train_dummy_model()
        with open(model_path, 'rb') as f:
            mood_model = pickle.load(f)
        print("Model successfully trained and saved.")
    else:
        with open(model_path, 'rb') as f:
            mood_model = pickle.load(f)
        print("Loaded existing model.")
    return mood_model

# Load the mood model
mood_model = load_or_regenerate_model(regenerate=True)

# Initialize Quadrant Predictor
try:
    quadrant_predictor = QuadrantPredictor(model_path='gaze_models/look_at_quadrants_model.pkl', scaler_path='gaze_models/scaler.pkl')
except FileNotFoundError as e:
    print(e)
    exit()

# Global variables
mood = "None"
playlist = []
tracking_thread = None
tracking_active = False

# Screen dimensions
# Dynamically get screen resolution using JavaScript and send to server if needed
SCREEN_WIDTH = 1920  # Replace with your actual screen width if known
SCREEN_HEIGHT = 1080  # Replace with your actual screen height if known

# Define test playlists with local MP3 files
# Define test playlists with local MP3 files
test_playlists = {
    'Happy': [
        {
            'name': 'Can’t Stop the Feeling - Justin Timberlake',     
            'file_path' : 'music/cant_stop_feeling.mp3',       
            'image_url': 'images/cant_stop_feeling.jpeg'
        },
        {
            'name': 'I Gotta Feeling - Black Eyed Peas',
            'file_path': 'music/i_gotta_feeling.mp3',
            'image_url': 'images/i_gotta_feeling.png'
        },
        {
            'name': 'Happy - Pharrell Williams',
            'file_path': 'music/happy_pharrell.mp3',
            'image_url': 'images/happy_pharrell.jpg'
        },
        {
            'name': 'Walking on Sunshine - Katrina and the Waves',
            'file_path': 'music/walking_on_sunshine.mp3',
            'image_url': 'images/walking_on_sunshine.png'
        },
        {
            'name': 'Happy Together - The turtles',
            'file_path': 'music/happy_together.mp3',
            'image_url': 'images/happy_together.png'
        },

    ],
    'Sad': [
        {
            'name': 'Someone Like You - Adele',
            'file_path': 'music/someone_like_you.mp3',
            'image_url': 'images/someone_like_you.png'
        },
        {
            'name': 'Fix You - Coldplay',
            'file_path': 'music/fix_you.mp3',
            'image_url': 'images/fix_you.jpg'
        },
        {
            'name': 'Let Her Go - Passenger',
            'file_path': 'music/let_her_go.mp3',
            'image_url': 'images/let_her_go.jpg'
        },
         {
            'name': 'The Night We Met - Lord Huton',
            'file_path': 'music/the_night_we_met.mp3',
            'image_url': 'images/the_night_we_met.png'
        },
         {
            'name': 'Tears in Heaven - Eric Clapton',
            'file_path': 'music/tears_in_heaven.mp3',
            'image_url': 'images/tears_in_heaven.png'
        },
    ],
    'Stressed': [
        {
            'name': 'Break Stuff - Limp Bizkit',
            'file_path': 'music/break_stuff.mp3',
            'image_url': 'images/break_stuff.jpeg'
        },
        {
            'name': 'Get Out My Way - Tedashii feat. Lecrae',
            'file_path': 'music/out_my_way.mp3',
            'image_url': 'images/out_my_way.jpeg'
        },
        {
            'name': 'Elevate - DJ Khalil',
            'file_path': 'music/elevate.mp3',
            'image_url': 'images/elevate.jpeg'
        },
        {
            'name': 'Smells Like Teen Spirit - Nirvana',
            'file_path': 'music/smells_like_teen_spirit.mp3',
            'image_url': 'images/smells_like_teen_spirit.png'
        },
        {
            'name': 'Killing in the Name - Rage Against the Machine',
            'file_path': 'music/killing_in_the_name.mp3',
            'image_url': 'images/killing_in_the_name.png'
        },
    ],
     'Relaxed': [
        {
            'name': 'Snowfall - Idealism',
            'file_path': 'music/snowfall.mp3',
            'image_url': 'images/snowfall.jpeg'
        },
        {
            'name': 'Moonlight Sonata - Ludwig van Beethoven',
            'file_path': 'music/moonlight_sonata.mp3',
            'image_url': 'images/moonlight_sonata.jpeg'
        },
        {
            'name': 'Weightless - Macroni Union',
            'file_path': 'music/weightless.mp3',
            'image_url': 'images/weightless.jpeg'
        },
        {
            'name': 'River Flows in You - Yiruma',
            'file_path': 'music/river_flows_in_you.mp3',
            'image_url': 'images/river_flows_in_you.png'
        },
        {
            'name': 'Clair de Lune - Claude Debussy',
            'file_path': 'music/clair_de_lune.mp3',
            'image_url': 'images/clair_de_lune.png'
        },
    ],
}


@app.route('/clear_session')
def clear_session():
    session.clear()
    return "Session cleared!", 200

# Function to map quadrants to screen coordinates
def map_quadrant_to_screen(quadrant):
    if quadrant == 'top_left':
        return 100, 100  # Adjust based on your preference
    elif quadrant == 'top_right':
        return SCREEN_WIDTH - 100, 100
    elif quadrant == 'bottom_left':
        return 100, SCREEN_HEIGHT - 100
    elif quadrant == 'bottom_right':
        return SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100
    else:
        return SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2  # Center

def tracking_function():
    global tracking_active
    tracking_active = True
    cap = cv2.VideoCapture(1)  # Change to 1 if your camera is on index 1
    if not cap.isOpened():
        print("Error: Unable to access camera.")
        tracking_active = False
        return

    print("Starting quadrant tracking...")
    while tracking_active:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Predict the quadrant and get the annotated frame
        quadrant, annotated_frame = quadrant_predictor.predict(frame)

        # If a quadrant is detected, emit the data
        if quadrant:
            screen_x, screen_y = map_quadrant_to_screen(quadrant)
            # Emit gaze coordinates via SocketIO
            socketio.emit('gaze', {'x': screen_x, 'y': screen_y, 'quadrant': quadrant})
            # print(f"Gaze Emitted: Quadrant={quadrant}, ScreenX={screen_x}, ScreenY={screen_y}")

        # Optional: Sleep to reduce CPU usage
        time.sleep(0.02)  # Approximately 50 FPS]

    # Release resources
    cap.release()
    tracking_active = False
    print("Quadrant tracking stopped.")

def start_tracking():
    global tracking_thread, tracking_active
    if not tracking_active:
        tracking_thread = threading.Thread(target=tracking_function)
        tracking_thread.daemon = True  # Ensure thread exits when main program does
        tracking_thread.start()
        print("Tracking thread started.")
    else:
        print("Tracking is already active.")

# Routes
# Start tracking automatically when the server starts
start_tracking()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/device_connection')
def device_connection():
    return render_template('device_connection.html')

@app.route('/detect_mood')
def detect_mood():
    global mood
    # will have to integrate this with the mood model backend.
    sample_data = pd.read_csv(os.path.join('static', 'data', 'sample_eeg_data.csv'))
    mood = mood_model.predict(sample_data)[0] 

    # Render detect_mood.html, which will proceed to show_mood after delay
    return render_template('detect_mood.html')

@app.route('/show_mood')
def show_mood():
    global mood
    if not mood:
        return redirect(url_for('detect_mood'))
    return render_template('mood.html', mood=mood.capitalize())


@app.route('/playlist')
def show_playlist():
    global mood
    if not mood:
        return redirect(url_for('detect_mood'))

    # Fetch playlist based on mood and initialize in session if not already set
    if 'playlist' not in session:
        playlist = test_playlists.get(mood.capitalize(), [])
        session['playlist'] = playlist
        session['current_song_index'] = 0
    else:
        playlist = session['playlist']

    current_song_index = session['current_song_index']

    # Check if playlist is empty or we reached the end
    if not playlist or current_song_index >= len(playlist):
        return redirect(url_for('thank_you'))

    current_song = playlist[current_song_index]

    # Debugging: Print the structure of the current song
    print(f"Serving Playlist: {current_song}")

    # Ensure current_song has the correct keys
    if not all(key in current_song for key in ['name', 'file_path', 'image_url']):
        return f"Error: Invalid song structure: {current_song}", 500

    return render_template('playlist.html', song=current_song, mood=mood.capitalize())

@app.route('/skip_song', methods=['POST'])
def skip_song():
    try:
        global mood
        print(f"Current mood: {mood}")  # Debug log
        
        if not mood:
            return jsonify({'success': False, 'message': 'Mood not detected.'}), 400

        # Fetch playlist from session
        playlist = session.get('playlist', [])
        current_song_index = session.get('current_song_index', 0)
        
        print(f"Playlist length: {len(playlist)}")  # Debug log
        print(f"Current index: {current_song_index}")  # Debug log

        # Validate playlist and index
        if not playlist:
            return jsonify({'success': False, 'message': 'Playlist is empty.'}), 400
        if current_song_index < 0 or current_song_index >= len(playlist):
            return jsonify({'success': False, 'message': 'Current song index is out of bounds.'}), 400

        # Move to the next song
        next_song_index = current_song_index + 1
        if next_song_index >= len(playlist):
            return jsonify({'success': False, 'redirect': True, 'redirect_url': url_for('thank_you')}), 200

        # Update session with the new current song index
        session['current_song_index'] = next_song_index

        try:
            next_song = playlist[next_song_index]
            print(f"Next song data: {next_song}")  # Debug log

            next_song_data = {
                'name': next_song['name'],
                'file_path': url_for('static', filename=next_song['file_path']),
                'image_url': url_for('static', filename=next_song['image_url']),
            }
        except IndexError as e:
            print(f"IndexError: {str(e)}")  # Log the specific error
            return jsonify({'success': False, 'message': 'Invalid song index.'}), 500
        except KeyError as e:
            print(f"KeyError: {str(e)}")  # Log missing keys
            return jsonify({'success': False, 'message': f"Missing song data: {str(e)}"}), 500

        return jsonify({
            'success': True,
            'song': next_song_data
        })

    except Exception as e:
        print(f"Error in skip_song: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'message': f"Server error: {str(e)}"
        }), 500

@app.route('/debug_state', methods=['GET'])
def debug_state():
    return jsonify({
        'mood': mood if 'mood' in globals() else None,
        'playlist_length': len(session.get('playlist', [])),
        'current_index': session.get('current_song_index', 0),
        'session_data': dict(session)
    })

@app.route('/rate_song', methods=['POST'])
def rate_song():
    data = request.get_json()
    action = data.get('action')
    
    if not action:
        return jsonify({'success': False, 'message': 'No action provided.'}), 400

    if action not in ['like', 'dislike']:
        return jsonify({'success': False, 'message': 'Invalid action.'}), 400

    # Fetch playlist from session
    playlist = session.get('playlist', [])
    current_song_index = session.get('current_song_index', 0)

    if action == 'like':
        # Handle like action (e.g., add to favorites)
        if action == 'like':
            # Handle like action (e.g., add to favorites)
            if playlist and current_song_index < len(playlist):
                print(f"User liked the song: {playlist[current_song_index]['name']}")
                # Return success message
                return jsonify({'success': True, 'message': 'Song liked!'}), 200
            else:
                return jsonify({'success': False, 'message': 'No song to like.'}), 400


    elif action == 'dislike':
        # Handle dislike action by moving to the next song
        next_song_index = current_song_index + 1
        if next_song_index >= len(playlist):
            return jsonify({'success': True, 'redirect': True, 'redirect_url': url_for('thank_you')}), 200

        # Update session with the new current song index
        session['current_song_index'] = next_song_index
        next_song = playlist[next_song_index]

        next_song_data = {
            'name': next_song['name'],
            'file_path': url_for('static', filename=next_song['file_path']),
            'image_url': url_for('static', filename=next_song['image_url']),
        }

        return jsonify({'success': True, 'song': next_song_data}), 200
    return '', 204

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('show_playlist'))

@app.route('/thankyou')
def thank_you():
    global tracking_active
    tracking_active = False
    return render_template('thankyou.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


# Define a function to handle termination signals
def graceful_exit(signum, frame):
    global tracking_active
    print("\nGracefully shutting down...")

    # Stop tracking if active
    if tracking_active:
        print("Stopping tracking...")
        tracking_active = False
        if tracking_thread and tracking_thread.is_alive():
            tracking_thread.join(timeout=2)  # Set timeout to 2 seconds

    # Stop the Flask-SocketIO server
    socketio.stop()
    print("Server stopped.")
    sys.exit(0)  # Exit the application

# Register the signal handlers
signal.signal(signal.SIGINT, graceful_exit)  # Handles Ctrl+C
signal.signal(signal.SIGTERM, graceful_exit) # Handles termination signal

if __name__ == '__main__':
    socketio.run(app, debug=True)