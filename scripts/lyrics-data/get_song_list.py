import requests
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
import pandas as pd

HEADERS = {
    "User-Agent": "BillboardLyrics/1.0 (https://github.com/example/cleanslate; contact@example.com) Python/requests"
}

@dataclass
class Song:
    title: str
    artists: set
    year: int
    wikipedia_link: str | None

def extract_year_end_chart(year: int) -> list[Song]:
    url = f"https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")
    
    songs = []
    table = soup.find("table", class_="wikitable")
    if not table:
        return songs
    
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) >= 3:
            title_cell = cells[1]
            artist_cell = cells[2]
            
            title_link = title_cell.find("a")
            if title_link:
                title = title_link.get_text(strip=True).strip('"')
                wiki_link = "https://en.wikipedia.org" + title_link.get("href") if title_link.get("href") else None
            else:
                title = title_cell.get_text(strip=True).strip('"')
                wiki_link = None
            
            artist_links = artist_cell.find_all("a")
            if artist_links:
                artists = {a.get_text(strip=True) for a in artist_links}
            else:
                artist_text = artist_cell.get_text(strip=True)
                artists = {a.strip() for a in re.split(r',|&|featuring|feat\.|and', artist_text, flags=re.IGNORECASE)}
            
            songs.append(Song(title=title, artists=artists, year=year, wikipedia_link=wiki_link))
    return songs

def collect_songs(start_year: int, end_year: int) -> list[Song]:
    # Map from normalized title -> list of songs with that title
    seen_songs: dict[str, list[Song]] = {}
    
    for year in range(start_year, end_year + 1):
        songs = extract_year_end_chart(year)
        for song in songs:
            normalized_title = song.title.lower().strip()
            normalized_artists = {a.lower() for a in song.artists}
            
            if normalized_title not in seen_songs:
                seen_songs[normalized_title] = [song]
            else:
                # Check if any existing song with this title has overlapping artists
                is_duplicate = False
                for existing_song in seen_songs[normalized_title]:
                    existing_artists = {a.lower() for a in existing_song.artists}
                    if normalized_artists & existing_artists:  # Set intersection - any overlap
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Same title but different artists - it's a different song
                    seen_songs[normalized_title].append(song)
    
    # Flatten the list of songs
    return [song for songs_list in seen_songs.values() for song in songs_list]

songs = collect_songs(1970, 2025)

df = pd.DataFrame([
    {
        "title": song.title,
        "artists": list(song.artists),
        "year": song.year,
        "wikipedia_link": song.wikipedia_link
    }
    for song in songs
])

output_path = "billboard_songs_1970_2025.parquet"
df.to_parquet(output_path, index=False, compression="gzip")
print(f"Saved {len(df)} songs to {output_path}")
