import BasicLayout from '@/layouts/BasicLayout';
import Dashboard from '@/pages/Dashboard';
import About from '@/pages/About';
import Setting from '@/pages/Setting';
import User from '@/pages/User';
import Visualizer from '@/pages/Visualizer';
import VersionTree from '@/pages/VersionTree';
import Jobs from '@/pages/Jobs';

const routerConfig = [
  {
    path: '/',
    component: BasicLayout,
    children: [
      { path: '/about', component: About },
      { path: '/user', component: User },
      { path: '/setting', component: Setting },
      { path: '/visualizer/:id', component: Visualizer },
      { path: '/', exact: true, component: Dashboard },
      { path: '/jobs', component: Jobs },
      { path: '/versiontree', component: VersionTree }
    ],
  },
];

export default routerConfig;
