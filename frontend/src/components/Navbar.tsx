import React from 'react';
import './Navbar.css';

const Navbar: React.FC = () => {
  return (
    <nav className="navbar">
      <div className="logo">Sketchers</div>
      <ul className="nav-list">
        <li>Home</li>
        <li>Contact</li>
      </ul>
    </nav>
  );
};

export default Navbar;