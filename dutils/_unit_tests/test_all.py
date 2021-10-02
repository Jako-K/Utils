# If True, tests things which are kinda annoying: plots (manually close), prints etc.
VERBOSE = True


import pandas as pd
import numpy as np
import os as os
from glob import glob as glob
import types as types
import cv2 as cv2
import unittest
import matplotlib.pyplot as plt
from dutils import input_output
import dutils as U

# Automatic key press to close windows
AUTO_PRESS = True # Will automatically close windows activated by VERBOSE=True
from pynput.keyboard import Key, Controller
keyboard = Controller()
if AUTO_PRESS:
    plt.ion() # Makes plt not block other commands, which enable `keyboard` to automatically close windows


# Print total line count
print("\n" * 2 + "_" * 70)
for (boolean, string) in [(False, "without spacing"), (True, "with spacing")]:
    test_count = input_output.get_line_count_file("./test_all.py", exclude_empty_lines=boolean)
    code_count, _ = input_output.get_line_counts_folder("../", only_extension=".py", exclude_empty_lines=boolean)
    print(f"Line counts ({string}):")
    print(f" - Tests: {test_count}\n"
          f" -  Code: {code_count}\n"
          f" - Total: {test_count + code_count}\n")


# Check that every module has correctly filled in its __all__ attribute. This is important cause `dutils.search` use it
public_modules = [var for var in dir(U) if var[0] != "_" and var != "search"]
not_accounted_for = []

for module_str in public_modules:
    module = getattr(U, module_str)
    in_all = module.__all__
    public_vars = [var for var in dir(module) if var[0] != "_"]

    for p_var in public_vars:
        if p_var not in in_all:
            not_accounted_for.append(module_str + "." + p_var)

if not_accounted_for:
    print("not accounted for in __all__s:")
    [print(" - ", var) for var in not_accounted_for]
else:
    print("No missing values in any __all__\n\n")



########################################################################################################################
##########################################             all_around                #######################################
########################################################################################################################


from dutils.all_around import *
class Test_all_around(unittest.TestCase):

    def test_assert_path(self):
        with self.assertRaises(TypeError): pandas_standardize_df("not a dataframe")
        norm_df = pandas_standardize_df(pd.DataFrame(np.array([1, 2, 3, 4])))
        self.assertEqual(str(norm_df), """          0\n0 -1.161895\n1 -0.387298\n2  0.387298\n3  1.161895""")


    def test_get_grid_coordinates(self):
        with self.assertRaises(TypeError): get_grid_coordinates(3, "2")
        with self.assertRaises(TypeError): get_grid_coordinates(None, 2)
        self.assertEqual(get_grid_coordinates(3,2), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])


    def test_sturges_rule(self):
        with self.assertRaises(TypeError): sturges_rule(123)
        with self.assertRaises(TypeError): sturges_rule(None)
        self.assertEqual(sturges_rule([1,2,3,4]), (1.004420127756955, 3))


    def test_unfair_coin_flip(self):
        with self.assertRaises(TypeError): unfair_coin_flip(123)
        with self.assertRaises(ValueError): unfair_coin_flip(0.0)
        with self.assertRaises(ValueError): unfair_coin_flip(1.0)


    def check_int_sign(self):
        with self.assertRaises(TypeError): int_sign(1.0)
        self.assertEqual(int_sign(-11230), -1)
        self.assertEqual(int_sign(11230), 1)


    def test_init_2d_list(self):
        with self.assertRaises(TypeError): init_2d_list(1, 2.0)
        with self.assertRaises(TypeError): init_2d_list(2.0, 1)

        self.assertEqual(
            init_2d_list(4, 3),
            [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        )


    def test_ndarray_to_bins(self):
        with self.assertRaises(TypeError): ndarray_to_bins("not array", 2)
        with self.assertRaises(ValueError): ndarray_to_bins(np.array([1, 2, 3, 4]), 0)

        array, bins, thresh = ndarray_to_bins(np.array([1, 2, 3, 4]), 2)
        self.assertEqual( list(array), [1, 1, 2, 3] )
        self.assertEqual( bins, 2)
        self.assertEqual( list(thresh), [1. , 2.5, 4. ])


########################################################################################################################
##########################################             Colors                ###########################################
########################################################################################################################


from dutils.colors import *
from dutils.colors import (_assert_type_str, _assert_color_scheme, _assert_color_word, _hex_to_rgb, _rgb_to_hex,
                           _legal_types, _scheme_name_to_colors)
class Test_colors(unittest.TestCase):

    def test_is_legal_hex(self):
        self.assertEqual(is_legal_hex("Something"), False)
        self.assertEqual(is_legal_hex([1,2,3]), False)
        self.assertEqual(is_legal_hex("#c51cbe"), True)


    def test_is_legal_rgb(self):
        self.assertEqual(is_legal_rgb([1, 1, 1, 1]), False)
        self.assertEqual(is_legal_rgb("#c51cbe"), False)
        self.assertEqual(is_legal_rgb((1, 2, 3)), True)
        self.assertEqual(is_legal_rgb([0, 0, 256]), False)
        self.assertEqual(is_legal_rgb([1, 2, 3]), True)
        self.assertEqual(is_legal_rgb([1, 2, 3.0]), False)


    def test_get_color_type(self):
        self.assertEqual(get_color_type("#c51cbe"), "hex")
        self.assertEqual(get_color_type("#c51cbe_asd"), None)
        self.assertEqual(get_color_type([0,0,255]), "rgb")
        self.assertEqual(get_color_type([0,0,255,0]), None)


    def test_assert_type_str(self):
        with self.assertRaises(ValueError): _assert_type_str("something")
        with self.assertRaises(ValueError): _assert_type_str("lalal")
        with self.assertRaises(TypeError): _assert_type_str(None)
        for color_type in _legal_types:
            _assert_type_str(color_type)


    def test_assert_color(self):
        with self.assertRaises(TypeError): assert_color("something")
        with self.assertRaises(TypeError): assert_color("lalal")
        with self.assertRaises(TypeError): assert_color([0, 23, 10.])
        with self.assertRaises(TypeError): assert_color("#c51cbe_")
        for color in ["#c51cbe", (1, 5, 100), [0, 23, 10]]:
            assert_color(color)


    def test_assert_color_scheme(self):
        with self.assertRaises(ValueError): _assert_color_scheme("not_seaborn")
        with self.assertRaises(TypeError): _assert_color_scheme(123)
        for scheme in _scheme_name_to_colors.keys():
            _assert_color_scheme(scheme)


    def test_assert_color_word(self):
        with self.assertRaises(TypeError): _assert_color_word( (0,0,0), "seaborn")
        with self.assertRaises(TypeError): _assert_color_word( "blue", None)
        with self.assertRaises(ValueError): _assert_color_word("not_a_color", "not_seaborn")

        for scheme, scheme_colors in _scheme_name_to_colors.items():
            for color in scheme_colors:
                _assert_color_word(color, scheme)


    def test_convert_color(self):
        with self.assertRaises(TypeError): convert_color("#ffffff", None)
        with self.assertRaises(TypeError): convert_color("#ffffff_", "rgb")
        with self.assertRaises(TypeError): convert_color("asd", "rgb")
        with self.assertRaises(TypeError): convert_color([1,2,3,4], "rgb")
        with self.assertRaises(TypeError): convert_color(np.array([1, 2, 3]), "hex")
        with self.assertRaises(TypeError): convert_color([1,2,3.0], "hex")
        with self.assertRaises(TypeError): convert_color([1,2,3,5], "hex")

        convert_color([1,2,3], "hex")
        convert_color((1,2,3), "hex")
        convert_color((1,2,3), "rgb")
        convert_color("#Ffffff", "hex")
        convert_color("#ffffff", "rgb")


    def test_random_color(self):
        with self.assertRaises(TypeError): random_color(None)
        with self.assertRaises(TypeError): random_color(amount=1.0, color_type="rgb")
        with self.assertRaises(TypeError): random_color(color_type="rgb", min_rgb=1.0)
        with self.assertRaises(TypeError): random_color(color_type="rgb", max_rgb=1.0)
        with self.assertRaises(ValueError): random_color(color_type="not_at_color_type_str")
        with self.assertRaises(ValueError): random_color(min_rgb=600)
        with self.assertRaises(ValueError): random_color(max_rgb=0)
        with self.assertRaises(ValueError): random_color(min_rgb=100, max_rgb=50)
        with self.assertRaises(ValueError): random_color(amount=0)

        self.assertEqual( len(random_color(amount=3, color_type="rgb")), 3)
        self.assertEqual( len(random_color(amount=3, color_type="hex")), 3)
        self.assertEqual( len(random_color(3, "hex", 120, 140)), 3)


    def test_hex_to_rgb(self):
        with self.assertRaises(ValueError): _hex_to_rgb("#fffff_")
        with self.assertRaises(TypeError): _hex_to_rgb([1,2,3])
        self.assertEqual(_hex_to_rgb("#ffffff"), (255,255,255))


    def test_rgb_to_hex(self):
        with self.assertRaises(TypeError): _rgb_to_hex("[1,2,3]")
        with self.assertRaises(ValueError): _rgb_to_hex([1,2,300])
        self.assertEqual(_rgb_to_hex((255, 255, 255)), "#ffffff")


    def test_display_colors(self):
        with self.assertRaises(TypeError): display_colors("asd")
        with self.assertRaises(TypeError): display_colors(["asd"])
        with self.assertRaises(TypeError): display_colors(["#fffff_"])
        with self.assertRaises(TypeError): display_colors([(1,2,3,4)])


        if VERBOSE:
            display_colors(list(_scheme_name_to_colors["seaborn"].values()))
            if AUTO_PRESS: keyboard.press("q")


    def test_get_color_scheme(self):

        with self.assertRaises(TypeError): get_color_scheme("seaborn", None)
        with self.assertRaises(ValueError): get_color_scheme("seaborn", "no_valid_type")

        self.assertEqual(get_color_scheme("seaborn"), list(_scheme_name_to_colors["seaborn"].values()))
        self.assertEqual(get_color_scheme("seaborn", "hex"), [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#e1ffff'
        ])


    def test_get_color(self):
        with self.assertRaises(TypeError): get_colors([1])
        with self.assertRaises(ValueError): get_colors([""])
        with self.assertRaises(ValueError): get_colors(["bluee"])
        with self.assertRaises(ValueError): get_colors(["blue", "green"], color_type="asd", detailed=True)

        with self.assertRaises(TypeError): get_color( (0, 0, 0))
        with self.assertRaises(TypeError): get_color("blue", None)
        with self.assertRaises(ValueError): get_color("blue", "not_a_valid_type", "seaborn")
        with self.assertRaises(ValueError): get_color("blue", "rgb", "not_seaborn")

        self.assertEqual(get_colors(["blue"]), [(31, 119, 180)])
        self.assertEqual(get_colors(["blue", "green"]), [(31, 119, 180), (44, 160, 44)])
        self.assertEqual(get_colors(["blue", "green"], color_type="hex"), ['#1f77b4', '#2ca02c'])
        self.assertEqual(get_colors(["blue", "green"], color_type="hex", detailed=True), {'blue': '#1f77b4', 'green': '#2ca02c'})


########################################################################################################################
##########################################             Formatting                #######################################
########################################################################################################################


from dutils.formatting import *
class Test_formatting(unittest.TestCase):

    def test_scientific_notation(self):
        with self.assertRaises(TypeError): scientific_notation("not a number", 2)
        with self.assertRaises(TypeError): scientific_notation(2, "not a number")
        with self.assertRaises(TypeError): scientific_notation(2, 2.0)

        self.assertEqual(scientific_notation(2.1234, 3), "2.123E+00")
        self.assertEqual(scientific_notation(10_000_000, 3), "1.000E+07")


    def test_string_to_dict(self):
        with self.assertRaises(TypeError): string_to_dict(123)
        self.assertEqual(string_to_dict("{'a':1, 'b':2}"), {'a': 1, 'b': 2})


    def test_string_to_list(self):
        with self.assertRaises(TypeError): string_to_list(123)
        self.assertEqual(string_to_list('[198, 86, 292, 149]'), ['198', '86', '292', '149'])
        self.assertEqual(string_to_list('[198, 86, 292, 149]', element_type=int), [198, 86, 292, 149])


########################################################################################################################
##########################################             Images                ###########################################
########################################################################################################################


#TODO add tests, all missing (remember to add test_image.png)
from dutils.images import *
class Test_images(unittest.TestCase):

    def test_load_images(self):
        # Grey
        self.assertEqual(len(load_image("./test_grey.png").shape), 2)
        # self.assertEqual(load_image("./test_grey.png", "rgb").shape[2], 3) # <-- Raises user warning
        self.assertEqual(len(load_image("./test_grey.png", "unchanged").shape), 2)

        # RGB
        self.assertEqual(load_image("./test_image.png", "rgb").shape[2], 3)
        self.assertEqual(len(load_image("./test_image.png", "grey").shape), 2)
        self.assertEqual(load_image("./test_image.png", "unchanged").shape[2], 3)

        # Alpha
        self.assertEqual(load_image("./test_alpha.png", "rgb").shape[2], 3)
        self.assertEqual(len(load_image("./test_alpha.png", "grey").shape), 2)
        self.assertEqual(load_image("./test_alpha.png", "unchanged").shape[2], 4)


    def test_rotate_images(self):
        img = cv2.imread("./dragon.jpg")
        self.assertEqual(cv2.rotate(img, cv2_rotate_map[90]).shape, (606, 465, 3))
        self.assertEqual(cv2.rotate(img, cv2_rotate_map[180]).shape, (465, 606, 3))
        self.assertEqual(cv2.rotate(img, cv2_rotate_map[270]).shape, (606, 465, 3))


    def test_assert_ndarray_image(self):
        grey = load_image("./test_grey.png")
        color = load_image("./test_image.png")
        alpha = load_image("./test_alpha.png")

        assert_ndarray_image(color)  # ok
        assert_ndarray_image(color, "color")  # ok
        assert_ndarray_image(alpha)  # ok
        assert_ndarray_image(alpha, "color")  # ok
        assert_ndarray_image(grey)  # ok
        assert_ndarray_image(grey, "grey")  # ok
        assert_ndarray_image(grey, "gray")  # ok
        assert_ndarray_image(color)  # ok
        assert_ndarray_image(grey)  # ok
        assert_ndarray_image(grey, "gray")  # ok
        assert_ndarray_image(alpha)  # ok
        assert_ndarray_image(alpha, "color")  # ok

        with self.assertRaises(ValueError): assert_ndarray_image(grey, "asd")
        with self.assertRaises(ValueError): assert_ndarray_image(grey.flatten())
        with self.assertRaises(ValueError): assert_ndarray_image(np.array([[123, 23], [14, -12]]), "gray")
        with self.assertRaises(TypeError): assert_ndarray_image(np.array([[123, 123], [12124, 12]]), "gray")
        with self.assertRaises(TypeError): assert_ndarray_image(color / 255., "grey")
        with self.assertRaises(ValueError): assert_ndarray_image(color, "grey")
        with self.assertRaises(ValueError): assert_ndarray_image(grey, "color")


    def test_show_hist(self):
        if VERBOSE:
            image = load_image("./dragon.jpg", load_type="rgb")
            show_hist(image)
            if AUTO_PRESS: keyboard.press("q")

            image = load_image("./dragon.jpg", load_type="grey")
            show_hist(image)
            if AUTO_PRESS: keyboard.press("q")


    def test_histogram_stretching(self):
        if VERBOSE:
            img = load_image("./dragon.jpg", "grey")
            img_new = histogram_stretching(img).astype(np.uint8)

            show_hist(img)
            if AUTO_PRESS: keyboard.press("q")
            show_hist(img_new)
            if AUTO_PRESS: keyboard.press("q")
            show_ndarray_image(img)
            if AUTO_PRESS: keyboard.press("q")
            show_ndarray_image(img_new)
            if AUTO_PRESS: keyboard.press("q")


    def test_gamma_correction(self):
        if VERBOSE:
            img = load_image("./dragon.jpg", "grey")
            show_ndarray_image(gamma_correction(img, 3.0))
            if AUTO_PRESS: keyboard.press("q")


########################################################################################################################
##########################################             imports                ##########################################
########################################################################################################################


from dutils.imports import *
from dutils import imports
class Test_imports(unittest.TestCase):
    def test_get_imports(self):
        with self.assertRaises(TypeError): get_imports("not a list")
        with self.assertRaises(ValueError): get_imports(["Not a valid request"])


    def test_get_module_path(self):
        with self.assertRaises(TypeError): get_imports("not a Module")
        self.assertEqual(os.path.abspath("../imports.py"), get_module_path(imports))


    def test_get_available_functions(self):
        with self.assertRaises(TypeError): get_imports("not a Module")
        self.assertEqual( sorted(get_available_functions(imports)), sorted(imports.__all__))


    def test_get_all_available_import_classes(self):
        with self.assertRaises(TypeError): get_imports("not a Module")
        self.assertEqual(get_all_available_import_classes(imports), [])


########################################################################################################################
##########################################             Input_output                #####################################
########################################################################################################################


# TODO add get_file_extension
from dutils.input_output import *
class Test_input_output(unittest.TestCase):

    def test_assert_path(self):
        with self.assertRaises(TypeError): assert_path(123)
        with self.assertRaises(ValueError): assert_path("./something_lalalaalala.txt.png")
        assert_path("../input_output.py")


    def test_assert_path_dont_exists(self):
        with self.assertRaises(TypeError): assert_path_dont_exists(123)
        with self.assertRaises(ValueError): assert_path_dont_exists("../input_output.py")
        assert_path_dont_exists("./something_lalalaalala.txt.png")


    def test_path_exists(self):
        with self.assertRaises(TypeError): path_exists(123)
        self.assertEqual(path_exists("../input_output.py"), True)
        self.assertEqual(path_exists("something_lalalaalala.txt.png"), False)


    def test_add_path_to_system(self):
        with self.assertRaises(TypeError): add_path_to_system(None)
        with self.assertRaises(ValueError): add_path_to_system("This_path_hopefully_dont_exists.asd")
        add_path_to_system("./")


    def test_get_current_directory(self):
        self.assertEqual(type(get_current_directory()), str)
        # Check if "input_output.py" is somewhere within the current folder
        paths = glob(os.path.join("../", "*"))
        self.assertEqual(sum([path.find("input_output.py") != -1 for path in paths]) != 0, True)


    def test_save_plt_plot(self):
        with self.assertRaises(TypeError): save_plt_plot(None)
        with self.assertRaises(TypeError): save_plt_plot("string", 123)
        with self.assertRaises(TypeError): save_plt_plot("string", None, 300.0)
        with self.assertRaises(ValueError): save_plt_plot("string.notJPG.", None, 300)

        # Test function work
        save_plt_plot("./test_plt.png")
        assert_path("./test_plt.png")
        os.remove("./test_plt.png")


    def test_get_file_basename(self):
        with self.assertRaises(TypeError): get_file_basename(123)
        with self.assertRaises(TypeError): get_file_basename("path", None)
        with self.assertRaises(ValueError): get_file_basename('C:/Users/JohnDoe/Desktop/test', assert_path_exists=True)

        self.assertEqual( get_file_basename('C:/Users/JohnDoe/Desktop/test.png', assert_path_exists=False), "test")
        self.assertEqual( get_file_basename('C:/Users/JohnDoe/Desktop/test', assert_path_exists=False), "test")
        self.assertEqual( get_file_basename('C:/Users/JohnDoe/Desktop/test.png', True, False), "test.png")
        self.assertEqual( get_file_basename('C:/Users/JohnDoe/Desktop/test.png.jpg', True, False), "test.png.jpg")


    def test_write_to_file(self):
        with self.assertRaises(TypeError): write_to_file(123, "string")
        with self.assertRaises(TypeError): write_to_file("string", 123)

        # Open file, write to it, check it's correct and delete it.
        with open("./test.txt", "w") as f: f.close()
        write_to_file("./test.txt", "hello_world 123")
        with open("./test.txt", "r") as F:
            self.assertEqual(F.read(), "hello_world 123")
        os.remove("./test.txt")


    def test_read_json(self):
        with self.assertRaises(TypeError): read_json(123)
        with self.assertRaises(ValueError): read_json("test.png")

        # Open file, write to it, check it's correct and delete it.
        with open("./test.json", "w") as f: f.close()
        write_to_file("./test.json", '{"hello_world": []}')
        with open("./test.json", "r") as F:
            self.assertEqual(F.read(), '{"hello_world": []}')
        os.remove("./test.json")


    def test_get_number_of_files(self):
        with self.assertRaises(TypeError): get_number_of_files(123)
        with self.assertRaises(ValueError): get_number_of_files("not_a_real_path.asdasd")

        # Create folder with 3 files, check number and delete all again.
        assert_path_dont_exists("./testdir")
        os.mkdir("./testdir")
        for i in range(3):
            with open(f"./testdir/t{i}.txt", "w") as f:
                f.close()
        self.assertEqual(get_number_of_files("./testdir"), 3)
        [os.remove(f"./testdir/t{i}.txt") for i in range(3)]
        os.rmdir("./testdir")


    def test_read_txt_file(self):
        with self.assertRaises(TypeError): read_file(123)

        # Open file, write to it, check it's correct and delete it.
        with open("./test.txt", "w") as f: f.close()
        write_to_file("./test.txt", "hello_world 123")
        self.assertEqual(read_file("./test.txt"), "hello_world 123")
        os.remove("./test.txt")


    def test_save_and_load_pickle(self):
        with self.assertRaises(TypeError): save_as_pickle(None, 123, "string")
        with self.assertRaises(ValueError): save_as_pickle("string.not_pkl", "string")

        with self.assertRaises(TypeError): load_pickle_file(12)
        with self.assertRaises(ValueError): load_pickle_file("string.not_pkl")

        save_as_pickle([1,2,3], "test.pkl", "./")
        self.assertEqual( load_pickle_file("./test.pkl"), [1,2,3])
        os.remove("./test.pkl")


    def test_copy_folder(self):
        os.mkdir("./testdir")
        with self.assertRaises(TypeError): copy_folder(123, "./testdir")
        with self.assertRaises(TypeError): copy_folder("./testdir", 123)
        with self.assertRaises(ValueError): copy_folder("./testdir", "./testdir")

        copy_folder("./testdir", "./testdir1")
        assert_path("./testdir1")

        os.rmdir("./testdir")
        os.rmdir("./testdir1")


    def test_is_folder(self):
        self.assertEqual(is_folder("./hello123", must_be_empty=True), False)
        os.mkdir("hello123")
        self.assertEqual(is_folder("./bad_folder_path"), False)
        self.assertEqual(is_folder("./test_all.py"), False)
        self.assertEqual(is_folder("./", must_be_empty=True), False)
        self.assertEqual(is_folder("./hello123", must_be_empty=True), True)
        self.assertEqual(is_folder("./hello123", must_be_empty=False), True)
        os.rmdir("./hello123")


    def test_make_file(self):
        make_file("./test_text0.txt", allow_override=True)
        assert_path("./test_text0.txt")
        make_file("./test_text0.txt", allow_override=True)
        with self.assertRaises(ValueError): make_file("./test_text0.txt", allow_override=False)
        os.remove("./test_text0.txt")


    def test_is_file(self):
        self.assertEqual(is_file("./test_text1.txt"), False)
        make_file("./test_text1.txt")
        self.assertEqual(is_file("./test_text1.txt"), True)
        with self.assertRaises(ValueError): is_file("./test_text1.txt", "txt")
        self.assertEqual(is_file("./test_text1.txt", ".txt"), True)
        self.assertEqual(is_file("./"), False)
        os.remove("./test_text1.txt")


    def test_remove_file(self):
        with self.assertRaises(ValueError): remove_file("./test_text2.txt")
        make_file("./test_text2.txt")
        remove_file("./test_text2.txt")
        assert_path_dont_exists("./test_text2.txt")
        with self.assertRaises(ValueError): remove_file("./")


    def test_file_count(self):
        os.mkdir("./file_count_test")
        make_file(f"./file_count_test/test.py")
        make_file(f"./file_count_test/test.txt")
        write_to_file(f"./file_count_test/test.py", "a\n" * 10 + " ")
        write_to_file(f"./file_count_test/test.txt", "a\n" * 10)

        with self.assertRaises(ValueError): get_line_count_file("./file_count_test")
        with self.assertRaises(ValueError): get_line_counts_folder(f"./file_count_test/test.py")

        self.assertEqual(get_line_count_file(f"./file_count_test/test.py"), 10)
        self.assertEqual( get_line_counts_folder(f"./file_count_test", exclude_empty_lines=False),
                          (21, {'test.py': 11, 'test.txt': 10}))
        self.assertEqual( get_line_counts_folder(f"./file_count_test", only_extension=".txt"),
                          (10, {'test.txt': 10}))

        remove_file(f"./file_count_test/test.py")
        remove_file(f"./file_count_test/test.txt")
        os.rmdir("./file_count_test")


########################################################################################################################
##########################################             jupyter                ##########################################
########################################################################################################################


from dutils.jupyter_ipython import *
class Test_jupyter(unittest.TestCase):
    def test_all(self):
        # I don't really know how to test these properly, since I cannot guarantee to be in a jupyter env.
        # I think the only real way is to test them in a manually in a jupyter env. every time i run tests

        # Run ./tests/test_jupyter.ipynb to test manually

        self.assertIn(in_jupyter(), [False, True])
        try:
            assert_in_jupyter()
        except RuntimeError:
            pass


########################################################################################################################
##########################################             pytorch                ##########################################
########################################################################################################################


# TODO add tests, all missing
from dutils.pytorch import *
class Test_pytorch(unittest.TestCase):
    def test_fold_performance_plot(self):
        if VERBOSE:
            data = np.array([[1, 2, 3, 4], [2, 1, 4, 5], [3, 2, 5, 1]])
            fold_performance_plot(data)
            if AUTO_PRESS: keyboard.press("q")


########################################################################################################################
######################################             system_info                ##########################################
########################################################################################################################


from dutils.system_info import *
class Test_system_info(unittest.TestCase):

    def test_get_vram_info(self):
        self.assertEqual( isinstance(windows_illegal_file_name_character, list), True)
        self.assertEqual( isinstance(get_vram_info(), dict) and len(get_vram_info()) == 4, True)


    def test_get_gpu_info(self):
        self.assertEqual(isinstance(get_gpu_info(), dict) and len(get_gpu_info()) == 5, True)


    def test_get_screen_dim(self):
        WxH = get_screen_dim(WxH=True)
        self.assertEqual(isinstance(WxH, tuple), True)
        self.assertEqual(isinstance(WxH[0], int), True)
        self.assertEqual(isinstance(WxH[1], int), True)

        HxW = get_screen_dim(WxH=False)
        self.assertEqual(isinstance(HxW, tuple), True)
        self.assertEqual(isinstance(HxW[0], int), True)
        self.assertEqual(isinstance(HxW[1], int), True)


    def test_get_os(self):
        self.assertEqual(isinstance(get_os(), str), True)


    def test_on_windows(self):
        self.assertEqual(isinstance(on_windows(), bool), True)


    def test_get_computer_info(self):
        import sys, io
        # Suppress print start
        suppress_text = io.StringIO()
        sys.stdout = suppress_text

        self.assertEqual(isinstance(get_computer_info(), type(None)), True)

        # Suppress print end
        sys.stdout = sys.__stdout__


########################################################################################################################
####################################             time_and_date                ##########################################
########################################################################################################################


from dutils.time_and_date import *
class UnitTests(unittest.TestCase):

    def test_stop_watch(self):
        # Init checks
        with self.assertRaises(TypeError):
            StopWatch(time_unit = "seconds", start_on_create= None, precision_decimals = 3)
        with self.assertRaises(TypeError):
            StopWatch(time_unit = "seconds", start_on_create= False, precision_decimals = 3.0)
        with self.assertRaises(TypeError):
            StopWatch(time_unit = None, start_on_create= False, precision_decimals = 3)
        with self.assertRaises(ValueError):
            StopWatch(time_unit = "seconds", start_on_create= False, precision_decimals = -1)

        for value in StopWatch().legal_units:
            StopWatch(time_unit=value)

        # Check all functions
        timer = StopWatch(precision_decimals=0)
        with self.assertRaises(RuntimeError): timer.stop()
        timer.start()
        timer.get_elapsed_time()
        timer.stop()
        timer.reset()
        timer.set_unit("minutes")


    def test_fps_timer(self):
        # Init checks
        with self.assertRaises(TypeError):
            FPSTimer(precision_decimals=None)
        with self.assertRaises(ValueError):
            FPSTimer(precision_decimals=-1)

        # Check all functions
        fps_timer = FPSTimer(precision_decimals=0)
        with self.assertRaises(RuntimeError): fps_timer.get_fps()
        with self.assertRaises(RuntimeError): fps_timer.increment()
        fps_timer.start()
        fps_timer.increment()
        self.assertEqual(fps_timer.get_frame_count(), 1)
        fps_timer.get_fps()
        fps_timer.increment()
        fps_timer.reset()


    def test_months(self):
        self.assertEqual(len(month_names) == len(month_names_abb) == 12, True)


########################################################################################################################
######################################             type_check                ###########################################
########################################################################################################################


from dutils.type_check import *
class Unit_type_check(unittest.TestCase):

    def test_assert_type(self):
        with self.assertRaises(TypeError):
            assert_type(to_check=0, expected_type=str)
        with self.assertRaises(ValueError):
            assert_type(to_check="string", expected_type=str, allow_none=22)
        assert_type(to_check=12, expected_type=int)
        assert_type(to_check=None, expected_type=type(None))
        assert_type(to_check=None, expected_type=str, allow_none=True)


    def test_assert_types(self):
        with self.assertRaises(TypeError):
            assert_types(to_check=[22, "string", None], expected_types=[int, str, int])
        with self.assertRaises(TypeError):
            assert_types(to_check=[22, "string", None], expected_types=[int, str, int])
        with self.assertRaises(TypeError):
            assert_types(to_check=["string", None], expected_types=[int, str])
        with self.assertRaises(ValueError):
            assert_types(to_check=[22, None], expected_types=[int, str, int], allow_nones=[0, 1, 0])
        with self.assertRaises(ValueError):
            assert_types(to_check=[22, None, 23], expected_types=[int, str, int], allow_nones=[0, 1])
        with self.assertRaises(TypeError):
            assert_types(to_check=[22, None], expected_types=[int, None], allow_nones=[0, 1])

        assert_types(to_check=[22, "string", None], expected_types=[int, str, int], allow_nones=[0, 0, 1])
        assert_types(
            to_check=[22, 0.2, None, unittest],
            expected_types=[int, float, str, types.ModuleType],
            allow_nones=[0, 0, 1, 0]
        )


    def test_assert_list_slow(self):
        with self.assertRaises(ValueError):
            assert_list_slow(to_check=[1, 2, 3], expected_type=int, expected_length=4)
        with self.assertRaises(TypeError):
            assert_list_slow(to_check=[1, 2, 3], expected_type=str)
        with self.assertRaises(TypeError):
            assert_list_slow(to_check=[None, "hello", 3], expected_type=str)
        with self.assertRaises(ValueError):
            assert_list_slow(to_check=[3, 4], expected_type=int, expected_length=-1)
        with self.assertRaises(TypeError):
            assert_list_slow(to_check=[1, "2", 3], expected_type=int)
        with self.assertRaises(TypeError):
            assert_list_slow(to_check=[1, None, 3], expected_type=int, allow_none=False)

        assert_list_slow(to_check=[1, None, 3], expected_type=int, allow_none=True)
        assert_list_slow(to_check=[1, 2, 3], expected_type=int)
        assert_list_slow(to_check=[None, None, None], expected_type=type(None))


    def test_assert_int(self):
        with self.assertRaises(RuntimeError): assert_in("a", 2)
        with self.assertRaises(ValueError): assert_in("a", ["b", "c"])
        assert_in("b", ["b", "c"])
        assert_in(1, ["b", 1, None])


    def test_assert_comparison_number(self):
        with self.assertRaises(TypeError): assert_comparison_number("3", 0, ">=", "number_of_cats")
        with self.assertRaises(TypeError): assert_comparison_number(3, 0, ">=", None)
        with self.assertRaises(TypeError): assert_comparison_number(3.0, 0, ">=", "number_of_cats")

        with self.assertRaises(ValueError): assert_comparison_number(3.0, 0.0, ">==", "number_of_cats")
        with self.assertRaises(ValueError): assert_comparison_number(3, 0, "==", "number_of_cats")
        with self.assertRaises(ValueError): assert_comparison_number(3, 0, "=", "number_of_cats")
        with self.assertRaises(ValueError): assert_comparison_number(3, 0, "<", "number_of_cats")
        with self.assertRaises(ValueError): assert_comparison_number(3., 0., "<=", "number_of_cats")
        with self.assertRaises(ValueError): assert_comparison_number(0., 3., ">", "number_of_cats")
        with self.assertRaises(ValueError): assert_comparison_number(0., 3., ">=", "number_of_cats")

        assert_comparison_number(3, 0, ">=", "number_of_cats")
        assert_comparison_number(3.0, 0.0, ">=", "number_of_cats")
        assert_comparison_number(3, 0, ">", "number_of_cats")
        assert_comparison_number(3.0, 3.0, "<=", "number_of_cats")
        assert_comparison_number(0.0, 3.0, "<", "number_of_cats")
        assert_comparison_number(0, 0, "==", "number_of_cats")
        assert_comparison_number(0, 0, "=", "number_of_cats")


if __name__ == "__main__":
    unittest.main(verbosity=1)

