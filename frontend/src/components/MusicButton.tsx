// src/components/MusicButton.tsx
import React, { useState, useRef } from 'react';
import './MusicButton.css';

interface MusicButtonProps {
  src: string; // path to the audio file
}

const MusicButton: React.FC<MusicButtonProps> = ({ src }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(new Audio(src));

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!isPlaying) {
      audio.play();
    } else {
      audio.pause();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <button className="music-button" onClick={togglePlay}>
      {isPlaying ? '🔊 Playing' : '🔇 Mute'}
    </button>
  );
};

export default MusicButton;