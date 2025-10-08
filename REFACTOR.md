# Ears Refactor Plan

Goal: Split the ears tool into a lightweight CLI frontend and a dedicated processing server.

Client (ears CLI):
	•	Handles only microphone input and audio streaming.
	•	Sends recorded audio in real time to the server using WebSockets (full-duplex over TCP).

Server (ears server):
	•	Receives audio stream.
	•	Runs speech-to-text inference using the Kyutai model or similar.
	•	Sends transcriptions back to the client over the same WebSocket connection.

Advantages:
	•	Decouples recording from transcription logic.
	•	Allows multiple clients to connect to the same server.
	•	Server can support multi-stream inference, processing multiple audio sources concurrently on a single GPU.

At the end we have 2 binarys:

ears

and 

ears-server
