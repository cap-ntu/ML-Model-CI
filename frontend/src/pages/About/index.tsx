import React from 'react';
import { Link, withRouter } from 'ice';

const About = () => {
  return (
    <div>
      <h1>About page</h1>
      <div><Link to="/">Home</Link></div>
    </div>
  );
};

export default About;