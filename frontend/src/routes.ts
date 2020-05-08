import BasicLayout from '@/layouts/BasicLayout';
import Dashboard from '@/pages/Dashboard';
import About from '@/pages/About';

const routerConfig = [
  {
    path: '/',
    component: BasicLayout,
    children: [
      {
        path: '/',
        exact: true,
        component: Dashboard,
      },
      {
        path: '/about',
        component: About,
      }
    ]
  }
];
export default routerConfig;
