import lyricsgenius
import os

def fetch_lyrics(api_token, artists=[
        "Linkin Park", 
        "Pink Floyd",
        "The Beatles",    
        "Nirvana",        
        "Metallica",    
        "The Doors"       

    ], max_songs_per_artist=30):
    genius = lyricsgenius.Genius(api_token, timeout=20, retries=3)
    genius.verbose = False  
    genius.remove_section_headers = True 
    
    corpus = ""
    
    for artist_name in artists:
        print(f"Fetching lyrics for {artist_name}")
        try:
            artist = genius.search_artist(artist_name, max_songs=max_songs_per_artist, sort="popularity")
            if artist:
                for song in artist.songs:
                    corpus += song.lyrics + "\n\n"
        except Exception as e:
            print(f"Error fetching {artist_name}: {e}")
    
    return corpus


def save_corpus(corpus, filename='music_corpus.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(corpus)
    print(f" Corpus saved")
    print(f"Approximate words: {len(corpus.split()):,}")


if __name__ == "__main__":
    api_token = os.getenv('GENIUS_API_TOKEN')
    
    if api_token:
        corpus = fetch_lyrics(api_token=api_token)
        save_corpus(corpus, 'music_corpus.txt')
    