"""Geometry, rigid grasp, swept path and physical sensor-chain acceptance."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from hard_disk_robot.adapters.mujoco import load_model, reset_home
from hard_disk_robot.adapters.mujoco.insertion_validation import solve_center, run_probe


class InsertionSceneTest(unittest.TestCase):
    def setUp(self):
        self.model, self.data = load_model()
        reset_home(self.model, self.data)

    def test_exact_geometry_and_interfaces(self):
        m, d = self.model, self.data
        np.testing.assert_allclose(m.geom('drive_collision').size*2, [.177,.106,.026], atol=1e-12)
        self.assertAlmostEqual(m.body('replacement_drive_held').mass[0], .2)
        expected_inertia = .2/12*np.array([.106**2+.026**2,.177**2+.026**2,.177**2+.106**2])
        np.testing.assert_allclose(m.body('replacement_drive_held').inertia, expected_inertia)
        width = 2*(m.geom('socket_left').pos[1]-m.geom('socket_left').size[1])
        height = 2*(m.geom('socket_top').pos[2]-m.geom('socket_top').size[2])
        np.testing.assert_allclose([width,height], [.1065,.0265], atol=1e-12)
        np.testing.assert_allclose((np.array([width,height])-[.106,.026])/2, [.00025,.00025])
        for name in ('left','right','top','bottom'):
            self.assertAlmostEqual(m.geom('socket_'+name).size[0]*2, .04)
            self.assertGreater(m.geom('socket_'+name).contype[0], 0)
        np.testing.assert_allclose(d.site('drive_center').xmat.reshape(3,3), np.eye(3), atol=1e-10)
        np.testing.assert_allclose(d.site('drive_front').xpos-d.site('socket_entry').xpos, [-.01,0,0], atol=1e-10)
        self.assertEqual(m.nu, 7)
        self.assertEqual(m.nsensordata, 6)
        names = [m.body(i).name for i in range(m.nbody)]
        self.assertFalse(any('server' in n or 'latch' in n or 'carrier' in n for n in names))
        # Approach from the rear 106 mm short edge, centered across its width.
        np.testing.assert_allclose(d.body('g_base').xmat.reshape(3,3)[:, 2], [1,0,0], atol=1e-10)
        np.testing.assert_allclose(d.site('pinch').xpos-d.site('drive_center').xpos,
                                   [-.0685,0,0], atol=1e-10)
        self.assertTrue(np.all(d.qpos[:6] >= m.jnt_range[:6,0]))
        self.assertTrue(np.all(d.qpos[:6] <= m.jnt_range[:6,1]))
        # Both physical inner pad planes touch the two thickness faces at init.
        base_rotation = d.body('g_base').xmat.reshape(3,3)
        for side, sign in [('right',1),('left',-1)]:
            for index in (1,2):
                g = m.geom(f'{side}_pad{index}').id
                local = base_rotation.T @ (d.geom_xpos[g]-d.body('g_base').xpos)
                self.assertAlmostEqual(sign*local[1]-m.geom_size[g,1], .013, places=9)

    def test_five_second_hold_and_repeatable_reset(self):
        m,d = self.model,self.data
        initial = d.qpos.copy()
        relative = d.body('g_base').xmat.reshape(3,3).T @ (d.site('drive_center').xpos-d.body('g_base').xpos)
        for i in range(round(5/m.opt.timestep)):
            mujoco.mj_step(m,d)
            self.assertEqual(d.ncon, 0)
            self.assertTrue(np.isfinite(d.sensordata).all())
        mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.site('drive_center').xpos, [.55,0,.45], atol=1e-6)
        np.testing.assert_allclose(d.body('g_base').xmat.reshape(3,3).T @ (d.site('drive_center').xpos-d.body('g_base').xpos), relative, atol=1e-12)
        self.assertTrue(np.isfinite(d.qpos).all())
        reset_home(m,d)
        np.testing.assert_array_equal(d.qpos,initial)

    def test_aligned_swept_path_to_50mm(self):
        m,d = self.model,self.data
        start = d.site('drive_center').xpos.copy()
        for advance in np.linspace(0,.06,121):
            d.qpos[:6] = solve_center(m,d.qpos[:6],start+[advance,0,0])
            mujoco.mj_forward(m,d)
            self.assertEqual(d.ncon,0, f'contact at advance={advance}: {list(d.contact)}')
        self.assertAlmostEqual(d.site('drive_front').xpos[0]-d.site('socket_entry').xpos[0],.05,places=8)

    def test_contact_and_half_timestep(self):
        for kind in ('axial','side'):
            normal, half = [run_probe(kind,dt) for dt in (.001,.0005)]
            self.assertTrue(normal['passed'],normal)
            self.assertTrue(half['passed'],half)
            np.testing.assert_allclose(normal['mean_tared_wrench'],half['mean_tared_wrench'],rtol=.1,atol=.02)


if __name__ == '__main__':
    unittest.main()
