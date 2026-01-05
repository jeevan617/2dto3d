import React from 'react';
import './About.css';

const About: React.FC = () => {
  return (
    <section id="about" className="about-section">
      <h2>About 3D World</h2>
      <p>
        Explore interactive 3D worlds with futuristic robots and immersive environments.
        Powered by Spline and React, this demo showcases real-time 3D rendering on the web.
      </p>
    </section>
  );
};

export default About; // <-- makes it a module