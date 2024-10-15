from __future__ import annotations

from typing import List
from pathlib import Path

import muspy
import music21


class ABCDataPreprocessor:

    def __init__(self, songs):
        self.songs = songs

    @classmethod
    def from_directory(cls, path: str) -> ABCDataPreprocessor:
        songs = ABCDataPreprocessor._read_songs(path)
        return cls(songs)

    @classmethod
    def from_muspy_repo(cls, path_to_save: str, name: str="nottingham_database") -> ABCDataPreprocessor:
        if name == "nottingham_database":
            dataset = muspy.NottinghamDatabase(path_to_save, download_and_extract=True)
        elif name == "esac":
            print("AAA")
            dataset = muspy.EssenFolkSongDatabase(path_to_save, download_and_extract=True)
        songs = ABCDataPreprocessor._read_songs(path_to_save + "/" + name)

        return cls(songs)
    
    @staticmethod
    def _read_songs(root_path: str) -> List[str]:
        skipped = 0
        root_path = Path(root_path)
        songs = []
        for file in (root_path).rglob("*.abc"):
            print(f"Processing file: {file}")
            try:
                songs += ABCDataPreprocessor._parse_abc_file(file)
            except Exception:
                skipped += 1
                print(f"{file} contains syntax error")
    
        print(f"Skipped {skipped} files due to syntax errors")
        return songs

    @staticmethod
    def _parse_abc_file(file: str) -> List[str]:
        with open(file, "r") as f:
            text = f.read()

        handler = music21.abcFormat.ABCHandler()
        handler.process(text)
        data = handler.splitByReferenceNumber()
        songs = []
        for song in data.values():
            song_text = "\n"
            for t in song.tokens:
                s = ""
                if isinstance(t, music21.abcFormat.ABCMetadata):
                    if t.isTempo() or t.isKey() or t.isMeter() or t.isDefaultNoteLength():
                        if song_text[-1] != "\n":
                            s += "\n"
                        s += f"{t.src}\n"
                else:
                    s += t.src
                song_text += t.stripComment(s)
            songs.append(song_text)

        return songs

    def export_to_file(self, path: str) -> None:
        text = ""
        for song in self.songs:
            if len(text) != 0 and text[-2:] != "\n\n":
                text += "\n\n"
            text += "X"
            if song[0] != "\n":
                text += "\n"
            text += song
        
        with open(path, "w") as f:
            f.write(text)
