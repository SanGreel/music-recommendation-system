# Content-based music recommendation system
In this repository is a music recommendation system based on track embeddings (representation).

We developed 2 approaches for embedding and search for similar audio compositions:
- Frequency decomposition
- Timbre and rhythm decomposition

Timbre and rhythm decomposition based approach shows impressive results. We think this algorithm can be used for real music recommend the system, at least as POC for this task.


## How to start?
If you are interested in some introduction to the music domain, we recommend You to run [audio_track_exploration.ipynb](https://github.com/SanGreel/music-recommendation-system/blob/master/audio_track_exploration.ipynb). In this file we describe some audio track characteristics and a few ways to decompose the audio track.


## How to reproduce? <br/>
### Frequency decomposition
Just run files from frequency decomposition folder:<br/><br/>
**0_create_collection.ipynb**<br/>
This script creates audio files database(collection) from the content of the 'data/audio' folder.<br/>

**1_audio_representation.ipynb**<br/>
Here we represent mp3 audio file as vectors for different frequencies. In our case, we processed only 20 seconds of the track.

**2_audio_recommendation.ipynb**<br/>
In this file, we calculate correlations between frequencies of the tracks and recommends the most relevant.
<br/><br/>
### Timbre and rhythm decomposition
Just run the file with instruction inside:
timbre_and_rhythm_features_decomposition/recommendation_system_usage.ipynb

## Links
We used some code from Music Information Retrieval Group - https://github.com/tuwien-musicir/rp_extract
