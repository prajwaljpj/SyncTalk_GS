# Recording Guide for Conversational Avatar Training

## What to Record (5-10 minutes total)

### Script Content Recommendations:

1. **Diverse Phonemes (2-3 minutes)**
   - Read pangrams: "The quick brown fox jumps over the lazy dog"
   - Include varied phonemes: "Peter Piper picked a peck of pickled peppers"
   - Different consonants: "She sells seashells by the seashore"
   - Vowel variations: "How now brown cow"

2. **Natural Speech Patterns (2-3 minutes)**
   - Introduce yourself naturally
   - Describe your interests, hobbies
   - Tell a short story or anecdote
   - Express different emotions (happy, thoughtful, excited, serious)

3. **Conversational Phrases (1-2 minutes)**
   - Common greetings: "Hello! How can I help you today?"
   - Questions: "What would you like to know?"
   - Transitions: "Let me think about that..." "That's a great question!"
   - Acknowledgments: "I understand", "Got it", "Makes sense"

4. **Varied Head Poses & Expressions**
   - Slight head movements (nodding, small turns)
   - Natural eye movements and blinking
   - Varied facial expressions (smile, neutral, thinking)
   - Different mouth openings (wide for laughing, small for 'mmm')

## Recording Checklist

- [ ] Camera mounted on tripod (stable, no shake)
- [ ] Face centered, fills 60-70% of frame
- [ ] Even, frontal lighting (no harsh shadows on face)
- [ ] Clean background
- [ ] Good audio quality (minimal echo/noise)
- [ ] Record at 25 FPS (or will need to convert)
- [ ] 512×512 resolution minimum
- [ ] Total duration 4-5 minutes minimum

## What to AVOID

- ❌ Hand movements in front of face
- ❌ Large head rotations (>45 degrees)
- ❌ Changing camera position mid-recording
- ❌ Backlighting (face should be well-lit)
- ❌ Moving background objects
- ❌ Long silences (keep talking!)
- ❌ Monotone delivery (vary your expression!)

## Sample Recording Script to Read

- "Please focus the camera on my face as we begin this recording session, specifically designed to capture every possible shape and movement of the human mouth. I want you to imagine a bright Monday morning where my brother Bob is baking a fresh batch of maple muffins in the kitchen. He carefully picks the biggest blueberries and places them into a large yellow bowl while humming a happy tune. Outside the window, five vivid violets are floating on the surface of the very vast river, and the view is absolutely beautiful. I thought about the thirty-three thieves who tried to seize the golden treasure from the royal zone, but they failed because the judge chose to check the gate quickly. It is funny how the smooth birch canoe slid on the planks near the dark blue background, while the white snow looked like silver sheets covering the ground. We saw three boys eating sweet cake near the sharp edge of the garden, and they were laughing at the lazy brown wolf who was trying to breathe through a thin cloth. Please leave the keys on the sleek steel sheet by the sheep, because the goose is loose and running wildly through the spruce grove. Who threw the blue shoe into the cool pool near the school? We may never know, but surely the vision of the beige garage is just a mirage. I am speaking slowly and distinctly so that the system can learn the difference between words like sheet and shoot, or fame and vain. Now, watch my jaw move closely as I count clearly to ten: one, two, three, four, five, six, seven, eight, nine, ten. The quick brown fox jumps over the lazy dog, and the sphinx judges my vows with great care. This concludes the video capture."

## Post-Recording

1. Verify video is 25 FPS:
   ```bash
   ffmpeg -i your_video.mp4
   # Look for "25 fps" in output
   ```

2. Convert to 25 FPS if needed:
   ```bash
   ffmpeg -i input.mp4 -filter:v fps=25 -c:v libx264 -crf 18 output_25fps.mp4
   ```

3. Crop to square if needed:
   ```bash
   ffmpeg -i input.mp4 -vf "crop=ih:ih" -c:v libx264 -crf 18 output_square.mp4
   ```

4. Resize to 512×512:
   ```bash
   ffmpeg -i input.mp4 -vf scale=512:512 -c:v libx264 -crf 18 output_512.mp4
   ```
