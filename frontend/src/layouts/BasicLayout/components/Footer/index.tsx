import React from 'react';
import styles from './index.module.scss';

export default function Footer() {
  return (
    <p className={styles.footer}>
      <span className={styles.logo}>CAP @NTU</span>
      <br />
      <span className={styles.copyright}>© 2019-2020 Nanyang Technological University, Singapore</span>
    </p>
  );
}