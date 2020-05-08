import * as React from 'react';
import styles from './index.module.scss';

const Pannel = () => {
  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Welcome to ModelCI!</h2>

      <p className={styles.description}>This is a awesome project, enjoy it!</p>

      <div className={styles.action}>
        
      </div>
    </div>
  );
};

export default Pannel;
