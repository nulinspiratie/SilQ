import unittest
import tempfile
import os

import silq
from silq.tools.config import *
from silq import config
from silq.pulses import Pulse

import qcodes as qc
from qcodes.instrument.parameter import Parameter


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2']}
        self.config = DictConfig('env1', folder=self.folder.name,
                                 config=self.d)

    def tearDown(self):
        self.folder.cleanup()

    def test_dicts_equal(self):
        self.assertTrue(self.dicts_equal(self.d, self.d))
        self.assertTrue(self.dicts_equal(self.d, self.config))
        self.assertTrue(self.dicts_equal(self.config, self.config))
        self.config.x = 2
        self.assertFalse(self.dicts_equal(self.d, self.config))

    def test_save_load_dir(self):
        self.config.save(folder=self.folder.name, save_as_dir=True)

        filepath = os.path.join(self.folder.name, f'{self.config.name}.json')
        folderpath = os.path.join(self.folder.name, self.config.name)
        self.assertFalse(os.path.exists(filepath))
        self.assertTrue(os.path.isdir(folderpath))
        for filename in self.config.keys():
            filepath = os.path.join(folderpath, '{}.json'.format(filename))
            self.assertTrue(os.path.exists(filepath))

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertFalse(config_loaded.save_as_dir)
        self.assertTrue(self.dicts_equal(self.config, config_loaded))

        new_folder = tempfile.TemporaryDirectory()
        config_loaded.pulses.save_as_dir = True
        config_loaded.save(folder=new_folder.name)

        os.path.isdir(os.path.join(new_folder.name, 'pulses'))
        os.path.isfile(os.path.join(new_folder.name, 'pulses', 'read.json'))
        os.path.isfile(os.path.join(new_folder.name, 'connections.json'))

        config_loaded2 = DictConfig('env1', folder=new_folder.name)
        self.assertTrue(self.dicts_equal(self.config, config_loaded2))
        self.assertIsInstance(config_loaded2.connections, ListConfig)
        new_folder.cleanup()

    def test_load_no_update(self):
        self.config.save(folder=self.folder.name, save_as_dir=True)
        self.config.pulses.pop('read')
        old_config = self.config.load(update=False)
        self.assertIn('read', old_config['pulses'])
        self.assertNotIn('read', self.config.pulses)
        self.assertEqual(old_config, self.d)
        self.assertNotEqual(self.config, self.d)

    def test_save_load_file(self):
        self.config.save(folder=self.folder.name)

        filepath = os.path.join(self.folder.name, f'{self.config.name}.json')
        folderpath = os.path.join(self.folder.name, self.config.name)
        self.assertTrue(os.path.exists(filepath))
        self.assertFalse(os.path.exists(folderpath))

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertFalse(config_loaded.save_as_dir)
        self.assertTrue(self.dicts_equal(self.config, config_loaded))

    def test_save_load_dependent(self):
        self.config.pulses.read2 = {'t_start': 1,
                                    't_stop': 'config:pulses.read.t_stop'}
        self.assertEqual(self.config.pulses.read2.t_stop,
                         self.config.pulses.read.t_stop)

        self.assertEqual(dict.__getitem__(self.config.pulses.read2, 't_stop'),
                         'config:pulses.read.t_stop')

        self.config.save(folder=self.folder.name)

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertEqual(config_loaded.pulses.read2.t_stop,
                         self.config.pulses.read.t_stop)
        self.assertEqual(dict.__getitem__(config_loaded.pulses.read2, 't_stop'),
                         'config:pulses.read.t_stop')

    def test_load_dependent(self):
        self.d = {
            't_start': 'config:pulses.read.t_start',
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2']}
        config = DictConfig('env1', folder=self.folder.name,
                                 config=self.d)

    def test_refresh_add_list_config(self):
        self.d['x'] = [1,2,3]
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config)
        self.assertEqual(self.config.x, self.d['x'])
        self.assertEqual(self.config, self.d)

    def test_refresh_add_list_subconfig(self):
        self.d['pulses']['x'] = [1,2,3]
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config.pulses)
        self.assertEqual(self.config.pulses.x, self.d['pulses']['x'])
        self.assertEqual(self.config, self.d)

    def test_refresh_add_dict_config(self):
        self.d['x'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config)
        self.assertEqual(self.config.x, self.d['x'])
        self.assertEqual(self.config, self.d)

    def test_refresh_add_dict_subconfig(self):
        self.d['pulses']['x'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config.pulses)
        self.assertEqual(self.config.pulses.x, self.d['pulses']['x'])
        self.assertEqual(self.config, self.d)

    def test_override_refresh(self):
        self.d['pulses']['read'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertIn('read', self.config.pulses)
        self.assertEqual(self.config, self.d)

    def test_override_refresh_remove_DictConfig(self):
        self.d['pulses'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertEqual(self.config, self.d)

    def test_refresh_exclusive_set(self):
        # Override dict setitem to record all items that are set
        dict_set_items = []
        dict_setitem = DictConfig.__setitem__
        def record_dict_setitem(self, key, val):
            dict_set_items.append((key, val))
            dict_setitem(self, key, val)
        DictConfig.__setitem__ = record_dict_setitem

        self.d['x'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertEqual(dict_set_items, [('x', {'test': 'val'}),
                                          ('test', 'val')])
        DictConfig.__setitem__ = dict_setitem

    def test_add_SubConfig(self):
        subconfig = DictConfig(name='pulses',
                               folder=None,
                               config={'read': {}})
        config.env1 = {'pulses': subconfig}
        self.assertIsInstance(config.env1, DictConfig)
        self.assertEqual(subconfig.config_path, 'config:env1.pulses')

    def test_add_ListConfig(self):
        config.env1 = [1,2,3]
        self.assertIsInstance(config.env1, ListConfig)

    def dicts_equal(self, d1, d2):
        d1_keys = list(d1.keys())
        for d2_key, d2_val in d2.items():
            if d2_key not in d1_keys:
                print(f'Key {d2_key} missing in d1')
                return False
            else:
                d1_keys.remove(d2_key)
                d1_val = d1[d2_key]
                if isinstance(d1_val, dict):
                    if not isinstance(d2_val, dict):
                        print(f'Key {d2_key} not dict in d2')
                        return False
                    if not self.dicts_equal(d1_val, d2_val):
                        return False
                elif not d1_val == d2_val:
                    print(f'Key {d2_key} differ ({d1_val} != {d2_val})')
                    return False
        return True


class TestConfigPath(unittest.TestCase):
    def setUp(self):
        self.original_environment = silq.environment
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2'],
            'properties': {},
            'env1': {
                'properties': {'x':1, 'y':2},
                'pulses': {
                    'read': {'t_start': 1}
                }
            }
        }
        self.config = DictConfig('env1', config=self.d)

    def tearDown(self):
        # Restore original environment
        silq.environment = self.original_environment

    def test_basic_config_paths(self):
        self.assertEqual(self.config.config_path, 'config:')
        self.assertEqual(self.config.pulses.config_path, 'config:pulses')
        self.assertEqual(self.config.pulses.read.config_path, 'config:pulses.read')
        self.assertEqual(self.config.connections.config_path, 'config:connections')

    def test_environment_config_paths(self):
        self.assertEqual(self.config.env1.config_path, 'config:env1')
        self.assertEqual(self.config.env1.properties.config_path,
                         'config:env1.properties')

        silq.environment = 'env1'
        self.assertEqual(self.config.env1.config_path, 'environment:')
        self.assertEqual(self.config.env1.properties.config_path,
                         'environment:properties')
        self.assertEqual(self.config.env1.pulses.config_path,
                         'environment:pulses')
        self.assertEqual(self.config.env1.pulses.read.config_path,
                         'environment:pulses.read')

    def test_get_config_path(self):
        self.assertEqual(self.config['config:'], self.config)
        self.assertEqual(self.config.pulses['config:'], self.config)
        self.assertEqual(self.config['config:properties'], self.config.properties)
        self.assertEqual(self.config.properties['config:pulses'], self.config.pulses)

    def test_get_environment_config_path(self):
        self.assertEqual(self.config['environment:'], self.config)
        self.assertEqual(self.config['environment:properties'],
                         self.config.properties)
        self.assertEqual(self.config.properties['environment:'], self.config)
        self.assertEqual(self.config.properties['environment:properties'],
                         self.config.properties)

        silq.environment = 'env1'
        self.assertEqual(self.config['environment:'], self.config.env1)
        self.assertEqual(self.config.properties['environment:'], self.config.env1)
        self.assertEqual(self.config['environment:pulses'], self.config.env1.pulses)
        self.assertEqual(self.config.properties['environment:pulses'], self.config.env1.pulses)


class TestConfigInheritance(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2']}
        self.config = DictConfig('env1', folder=self.folder.name,
                                 config=self.d)

    def tearDown(self):
        self.folder.cleanup()

    def test_inherit_relative(self):
        self.config.pulses.read2 = {'inherit': 'read'}
        self.assertEqual(self.config.pulses.read2.inherit, 'read')
        self.assertIn('t_start', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_start,
                         self.config.pulses.read.t_start)
        self.assertIn('t_stop', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_stop,
                         self.config.pulses.read.t_stop)

    def test_inherit_absolute(self):
        self.config.pulses2 = {'read':
                                   {'t_start': 1,
                                    't_stop': 21}}
        self.config.pulses.read2 = {'inherit': 'config:pulses2.read'}
        self.assertEqual(self.config.pulses.read2.inherit, 'config:pulses2.read')
        self.assertEqual(self.config[self.config.pulses.read2.inherit],
                         self.config.pulses2.read)
        self.assertIn('t_start', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_start,
                         self.config.pulses2.read.t_start)
        self.assertIn('t_stop', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_stop,
                         self.config.pulses2.read.t_stop)

    def test_override_inherit(self):
        self.config.pulses.read2 = {'inherit': 'read', 't_stop': 20}
        self.assertEqual(self.config.pulses.read2.inherit, 'read')
        self.assertIn('t_start', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_start,
                         self.config.pulses.read.t_start)
        self.assertIn('t_stop', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_stop, 20)


class TestConfigEnvironment(unittest.TestCase):
    def setUp(self):
        self.original_environment = silq.environment
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2'],
            'properties': {},
            'env1': {'properties': {'x': 1, 'y': 2}}}
        self.config = DictConfig('config', config=self.d)

    def tearDown(self):
        # Restore original environment
        silq.environment = self.original_environment

    def test_no_environment(self):
        silq.environment = None
        self.assertEqual(self.config.config_path, 'config:')
        self.assertEqual(self.config.properties.config_path, 'config:properties')
        self.assertEqual(self.config.env1.properties.config_path, 'config:env1.properties')

    def test_other_environment(self):
        silq.environment = 'env2'
        self.assertEqual(self.config.config_path, 'config:')
        self.assertEqual(self.config.properties.config_path, 'config:properties')
        self.assertEqual(self.config.env1.properties.config_path, 'config:env1.properties')

    def test_environment(self):
        silq.environment = 'env1'
        self.assertEqual(self.config.config_path, 'config:')
        self.assertEqual(self.config.properties.config_path, 'config:properties')
        self.assertEqual(self.config.env1.properties.config_path, 'environment:properties')

    def test_switch_environments(self):
        silq.environment = 'env2'
        self.assertEqual(self.config.config_path, 'config:')
        self.assertEqual(self.config.properties.config_path, 'config:properties')
        self.assertEqual(self.config.env1.properties.config_path, 'config:env1.properties')
        silq.environment = 'env1'
        self.assertEqual(self.config.config_path, 'config:')
        self.assertEqual(self.config.properties.config_path, 'config:properties')
        self.assertEqual(self.config.env1.properties.config_path, 'environment:properties')
        silq.environment = None
        self.assertEqual(self.config.config_path, 'config:')
        self.assertEqual(self.config.properties.config_path, 'config:properties')
        self.assertEqual(self.config.env1.properties.config_path, 'config:env1.properties')


class TestConfigMirroring(unittest.TestCase):
    def setUp(self):
        self.original_environment = silq.environment
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2'],
            'properties': {},
            'env1': {'properties': {'x': 1, 'y': 2}}}
        self.config = DictConfig('config', config=self.d)

    def tearDown(self):
        # Restore original environment
        silq.environment = self.original_environment

    def test_simple_mirroring(self):
        silq.environment = None

        with self.assertRaises(KeyError):
            self.config.properties.x = 'config:properties.y'

        self.config.properties.y = 1
        self.config.properties.x = 'config:properties.y'
        self.assertEqual(self.config.properties.y, 1)
        self.assertEqual(self.config.properties.x, 1)

        self.config.properties.x = 2
        self.assertEqual(self.config.properties.y, 1)
        self.assertEqual(self.config.properties.x, 2)

        self.config.properties.y = 3
        self.assertEqual(self.config.properties.y, 3)
        self.assertEqual(self.config.properties.x, 2)

    def test_chained_mirroring(self):
        self.config.properties.z = 1
        self.config.properties.y = 'config:properties.z'
        self.config.properties.x = 'config:properties.z'
        self.config.properties.w = 'config:properties.y'

        self.assertEqual(self.config.properties.z, 1)
        self.assertEqual(self.config.properties.y, 1)
        self.assertEqual(self.config.properties.x, 1)
        self.assertEqual(self.config.properties.w, 1)

        self.config.properties.z = 2
        self.assertEqual(self.config.properties.z, 2)
        self.assertEqual(self.config.properties.y, 2)
        self.assertEqual(self.config.properties.x, 2)
        self.assertEqual(self.config.properties.w, 2)

    def test_mirroring_change_environment(self):
        silq.environment = 'env1'
        self.config.env1.properties.y = 'environment:properties.x'
        self.assertEqual(self.config.env1.properties.y, 1)

        silq.environment = None
        with self.assertRaises(AttributeError):
            self.config.env1.properties.y


class TestConfigSignals(unittest.TestCase):
    def setUp(self):
        self.original_environment = silq.environment
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2'],
            'properties': {}}
        self.config = DictConfig('env1', config=self.d)

        self.emitted_signals = []
        self.config.signal.connect(self.register_signal)

    def tearDown(self):
        # Restore original environment
        silq.environment = self.original_environment

    def register_signal(self, sender, value):
        self.emitted_signals.append((sender, value))

    def test_simple_signal_no_environment(self):
        self.config.properties.x = 1
        self.assertEqual(len(self.emitted_signals), 2)
        self.assertEqual(self.emitted_signals[0], ('config:properties.x', 1))
        self.assertEqual(self.emitted_signals[1], ('environment:properties.x', 1))

    def test_simple_signal_environment(self):
        silq.environment = 'env1'
        self.config.properties.x = 1
        self.assertEqual(len(self.emitted_signals), 1)
        self.assertEqual(self.emitted_signals[0], ('config:properties.x', 1))

        self.config.env1 = {'properties.y': 42}
        self.emitted_signals = []

        self.config.env1.properties.y = 43
        self.assertEqual(len(self.emitted_signals), 1)
        self.assertEqual(self.emitted_signals[0], ('environment:properties.y', 43))

    def test_signal_mirroring(self):
        with self.assertRaises(KeyError):
            self.config.properties.x = 'config:properties.y'

        self.config.properties.y = 1
        self.config.properties.x = 'config:properties.y'
        self.assertEqual(self.config.properties.x, 1)

        self.emitted_signals = []
        self.config.properties.y = 2
        self.assertEqual(len(self.emitted_signals), 4)
        self.assertEqual(self.config.properties.y, 2)
        self.assertEqual(self.config.properties.x, 2)
        self.assertEqual(self.emitted_signals[0], ('config:properties.y', 2))
        self.assertEqual(self.emitted_signals[1], ('environment:properties.y', 2))
        self.assertEqual(self.emitted_signals[2], ('config:properties.x', 2))
        self.assertEqual(self.emitted_signals[3], ('environment:properties.x', 2))

    def test_signal_mirroring_environment(self):
        self.config.env1 = {'properties': {}}
        silq.environment = 'env1'

        self.config.env1.properties.y = 1
        self.config.env1.properties.x = 'environment:properties.y'
        self.assertEqual(self.config.env1.properties.x, 1)

        self.emitted_signals = []
        self.config.env1.properties.y = 2
        self.assertEqual(self.config.env1.properties.y, 2)
        self.assertEqual(self.config.env1.properties.x, 2)
        self.assertEqual(self.emitted_signals[0], ('environment:properties.y', 2))
        self.assertEqual(self.emitted_signals[1], ('environment:properties.x', 2))

    def test_signal_inheritance(self):
        silq.environment = 'env1'
        self.config.properties.x = 1
        self.assertEqual(len(self.emitted_signals), 1)
        self.assertEqual(self.emitted_signals[0], ('config:properties.x', 1))

        self.config.properties2 = {'inherit': 'config:properties'}
        self.assertEqual(self.config.properties._inherited_configs, ['config:properties2'])
        self.assertIn('x', self.config.properties2)
        self.assertEqual(self.config.properties2.x, 1)
        # No signal emitted
        self.assertEqual(len(self.emitted_signals), 3)
        self.assertEqual(self.emitted_signals[1],
                         ('config:properties2.inherit', 'config:properties'))
        self.assertEqual(self.emitted_signals[2],
                         ('config:properties2', {'inherit': 'config:properties'}))

        self.config.properties.x = 2
        self.assertEqual(len(self.emitted_signals), 5)
        self.assertEqual(self.emitted_signals[3], ('config:properties.x', 2))
        self.assertEqual(self.emitted_signals[4], ('config:properties2.x', 2))

    def test_signal_relative_inheritance(self):
        silq.environment = 'env1'
        self.config.pulses.read2 = {'inherit': 'read'}
        self.assertEqual(self.config.pulses.read._inherited_configs,
                         ['config:pulses.read2'])

        self.emitted_signals = []
        self.config.pulses.read.t_start = 30
        self.assertEqual(self.config.pulses.read.t_start, 30)
        self.assertEqual(self.config.pulses.read2.t_start, 30)

        self.assertEqual(len(self.emitted_signals), 2)
        self.assertEqual(self.emitted_signals[0],
                         ('config:pulses.read.t_start', 30))
        self.assertEqual(self.emitted_signals[1],
                         ('config:pulses.read2.t_start', 30))

    def test_refresh_signals(self):
        silq.environment = 'env1'
        self.d['properties']['x'] = 2
        self.config.refresh(self.d)
        self.assertEqual(self.config.properties.x, 2)
        self.assertEqual(len(self.emitted_signals), 1)
        self.assertEqual(self.emitted_signals[0], ('config:properties.x', 2))

        self.d.pop('properties')
        self.config.refresh(self.d)
        self.assertNotIn('properties', self.config)
        self.assertEqual(len(self.emitted_signals), 1)

        self.d['pulses']['read2'] = {'t_start': 1,
                                     't_stop': 2}
        self.config.refresh(self.d)
        self.assertEqual(len(self.emitted_signals), 4)
        self.assertEqual(self.emitted_signals[1], ('config:pulses.read2.t_start', 1))
        self.assertEqual(self.emitted_signals[2], ('config:pulses.read2.t_stop', 2))
        self.assertEqual(self.emitted_signals[3], ('config:pulses.read2',
                                                   {'t_start': 1,
                                                    't_stop': 2}))


class test_connect_parameter_to_config(unittest.TestCase):
    def setUp(self):
        self.original_environment = silq.environment
        self.silq_config = silq.config

        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2'],
            'properties': {},
            'env1': {'properties': {'x': 1, 'y': 2}}}
        self.config = DictConfig('cfg', config=self.d)
        qc.config.user.silq_config = silq.config = self.config

        self.emitted_signals = []
        self.config.signal.connect(self.register_signal)

    def tearDown(self):
        silq.environment = self.original_environment
        qc.config.user.silq_config = silq.config = self.silq_config

    def register_signal(self, sender, value):
        self.emitted_signals.append((sender, value))

    def test_simple_connect_parameter(self):
        p = Parameter(name='param1', config_link='config:properties.x',
                      set_cmd=None, initial_value=42)

        silq.config.properties.x = 41
        self.assertEqual(p(), 41)

        def test_connect_parameter_to_environment(self):
            p = Parameter(name='param1', config_link='config:properties.x',
                          set_cmd=None, initial_value=42)

            silq.config.properties.x = 41
            self.assertEqual(p(), 41)

    def test_connect_parameter_to_environment_config(self):
        p = Parameter(name='param1', config_link='environment:properties.x',
                      set_cmd=None, initial_value=42)
        silq.config.env1.properties.x = 41
        self.assertEqual(p(), 42)

        silq.config.properties.x = 43
        self.assertEqual(p(), 43)

        silq.environment = 'env1'
        silq.config.env1.properties.x = 44
        self.assertEqual(p(), 44)

        silq.config.properties.x = 45
        self.assertEqual(p(), 44)

        silq.environment = None
        silq.config.env1.properties.x = 46
        self.assertEqual(p(), 44)

        silq.config.properties.x = 47
        self.assertEqual(p(), 47)


if __name__ == '__main__':
    unittest.main()
