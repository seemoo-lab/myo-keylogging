# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2019-2021  Matthias Gazzari, Annemarie Mattmann
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

code="en"


for_science = """
                            _____      ______      ___ 
                           /   __`\   / ____ \    /\  `----_ 
                          /\   \_/   /\ \__/\ \   \ \   ____\ 
                         /\__    _\  \ \ \ \ \ \   \ \  \___/ 
                         \/_/\   \/   \ \ \_\_/ \   \ \  \ 
                            \ \___\    \ \______/    \ \__\ 
                             \/___/     \/_____/      \/__/ 

  ______       _______     ____       _______     ___            _______      _______      ____ 
 / _____\     /   ____\   /\___\     /  ____ \   /\  `---_      /   ____\    /  ____ \    /\   \ 
/\ \____/_   /\  /____/   \/___/_   /\  \__/\ \  \ \   __ `\   /\  /____/   /\  \__/\ \   \ \   \ 
\ \_____  \  \ \ \_         /\   \  \ \  _____/   \ \  \/\  \  \ \ \_       \ \  _____/    \ \___\ 
 \/_____\  \  \ \_ \______  \ \   \  \ \ \___/__   \ \  \ \  \  \ \_ \______ \ \ \___/__    \/___/_ 
  /\_______/   \_ \_______\  \ \___\  \ \_______\   \ \__\ \__\  \_ \_______\ \ \_______\     /\___\ 
  \/______/      \/_______/   \/___/   \/_______/    \/__/\/__/    \/_______/  \/_______/     \/___/ 
"""


thank_you_cat = """
   ,   , 
  //\,//\ 
 / ,  , |    ,--. 
(  *  * )   / ,-' 
 == .1.==._( ( 
 /         `. \ 
|         _  \ \ 
 \ \ ,   /     / 
  || |-__\_   / 
 ((_/  (___,-' 
"""


# define left and right key characters
# english layout
left_chars = ["1", "2", "3", "4", "5", "q", "w", "e", "r", "t", "a", "s", "d", "f", "g", "z", "x", "c", "v", "b"]
left_upper_chars = ["!", "@", "#", "$", "%"] + [el.upper() for el in left_chars[5:]]
right_chars = ["6", "7", "8", "9", "0", "-", "=", "y", "u", "i", "o", "p", "[", "]", "h", "j", "k", "l", ";", "'", "n", "m", ",", ".", "/"]
right_upper_chars = ["^", "&", "*", "(", ")", "_", "+"] + [el.upper() for el in right_chars[7:12]] + ["{", "}"] + [el.upper() for el in right_chars[14:18]] + [":", '"'] + [el.upper() for el in right_chars[20:22]] + ["<", ">", "?"]

# pair left and right key characters with their shift-modified version
left_keys = tuple(zip(left_chars, left_upper_chars))
right_keys = tuple(zip(right_chars, right_upper_chars))
keys = tuple(left_keys + right_keys)

# common english layout (same keys as other layouts)
common_left_chars = left_chars
common_left_upper_chars = left_upper_chars
common_right_chars = right_chars.remove("=")
common_right_upper_chars = right_upper_chars.remove("+")
common_left_keys = tuple(zip(left_chars, left_upper_chars))
common_right_keys = tuple(zip(right_chars, right_upper_chars))
common_keys = tuple(left_keys + right_keys)

# define key codes
left_codes = [10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 38, 39, 40, 41, 42, 50, 52, 53, 54, 55, 56]
right_codes = [15, 16, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32, 33, 34, 35, 36, 43, 44, 45, 46, 47, 48, 57, 58, 59, 60, 61, 62]
space_code = [65]
keycodes = left_codes + right_codes + space_code

# helper texts for tasks
# input
continue_ready_prompt = "(If you are ready, please press RETURN once to continue...)"
continue_prompt = "(Please press RETURN once to continue...)"
start_prompt = "(Please press RETURN once to start...)"
continue_or_stay_prompt = "(Please press RETURN again to continue or any other key to stay...)"
repeat_prompt = "(If something went wrong and you want to repeat the last task, press \"r\"...)"

# initial
intro_prompt = """Introduction\n
In the following, you will be presented with a series of tasks. Please read the instructions carefully and follow them.
Task types will repeat. You can go by the titles and skip reading the instructions if you remember them.
There will be no recording in between tasks, which is when the instructions will offer you to take a break unless you want to continue immediately.
Should you feel overwhelmed, should you need a break or should you want to stop, do not hesitate to say so.
Should something go wrong, we can restart the recording at any task.\n
""" + start_prompt

# sync
sync_prompt = """Please clap your hands once.\n
As soon as a hint on how to continue pops up, the clap was registered and you can move on.
If no hint pops up, please try again (the movement needs to be distinct but must not hurt, the loudness is not relevant).
We need to record this because it creates a sudden stop movement of both of your hands with which we can synchronize the two Myos.
Thank you.\n
""" + continue_prompt

# before task
start_uniform_random_memory_task_prompt = """Disappearing Random Characters\n
In the following, you will be presented with a few sets of characters to copy which will disappear as soon as you hit a key.
Please memorize them carefully before you start typing.
Please note that for this task return is shown as {} and space as {}.
The last set of characters may be smaller than those before.
Your input will be processed automatically.\n
{}""".format(chr(8629), chr(9251), start_prompt)
start_uniform_random_task_prompt = """Lasting Random Characters\n
In the following, you will be presented with a few sets of characters to copy.
Please note that for this task return is shown as {} and space as {}.
The last set of characters may be smaller than those before.
Your input will be processed automatically.\n
{}""".format(chr(8629), chr(9251), start_prompt)
start_copy_text_prompt = """Text\n
In the following, you will be given a few texts to copy.
Your input will be processed as soon as you hit RETURN twice.
Please use return only once you finished copying the text - it will be wrapped automatically.\n
""" + start_prompt
start_copy_pangram_prompt = """Pangrams\n
In the following, you will be given a few pangrams to copy (mostly senseless sentences in which every letter appears at least once).
Your input will be processed as soon as you hit RETURN twice.\n
""" + start_prompt
start_game_prompt = """Game\n
The last task will be a game, which will be explained in the following.
At the end of the game tutorial there will be an ingame snapshot.
Navigate through the tutorial and start the game by pressing RETURN.
You can choose yourself when to stop - there is no determined ending.
Have fun!\n
""" + start_prompt

# during task
copy_prompt = "Please copy the below:"
copy_input_prompt = "Please copy the below and hit RETURN:"

# between tasks
end_task_prompt = """Thank you!
You may take a break now, until you are ready for the next task.\n
""" + continue_ready_prompt + "\n" + repeat_prompt
end_task_with_posture_change_prompt = """Please change your posture (sitting position, hand position, keyboard position and/or body posture).\n
Thank you!
You may take a break now, until you are ready for the next task.\n
""" + continue_ready_prompt + "\n" + repeat_prompt
end_task_with_extended_break_prompt = """Thank you!
How about a small break? Just get up, walk around a bit, stretch.\n
""" + continue_ready_prompt + "\n" + repeat_prompt

# after tasks
other_task = "Recording stopped at task "
other_task_cont = "To continue with the next task, call the program with the same arguments and include -c "
last_task = "This was the last task.\n"
outro_prompt = "Many thanks for taking the time to support us in our research!" + thank_you_cat + "\n"


# pw task specific texts
intro_pw_prompt = """Introduction\n
In the following, you will be presented with a series of tasks in which you learn multiple \"passwords\" by repeatedly typing the respective keys. We want to use this data to see how well our machine learning models - trained to detect and identify key presses within the data recorded by the Myo - can infer passwords.
Please read the instructions carefully and follow them.
The task description will repeat for each task. If you remember it, you can skip reading it.
There will be no recording in between tasks, which is when the instructions will offer you to take a break unless you want to continue immediately.
Should you feel overwhelmed, should you need a break or should you want to stop, do not hesitate to say so.
Should something go wrong, we can restart the recording at any task.\n
""" + start_prompt
random_task = "Random "
pwgen_task = "pwgen-Generated "
xkcd_task = "xkcd-Style "
insecure_task = "Insecure "
start_pw_task_prompt = """Passwords\n
In the following, you will be presented with a few sets of characters to copy multiple times.
Please note that for this task space is shown as {}.
Please hit return once you have finished typing a password. The program will show whether the password is correct.
Hit return again to continue on.\n
{}""".format(chr(9251), start_prompt)
copy_pw_prompt = "Please type the below password (given in bold font):"
additional_pw_prompt = "(datastudy) user% continue\n Continuing requires a password..." + for_science
task_changes = "Nothing special here...\n\n" + continue_prompt
task_changes_coin = "Found an orb!\n\n" + continue_prompt
task_changes_extra_coin = "Found a chest.\nImagine a long cutscene is playing in which you open it...\nFound three orbs!\n\n" + continue_prompt
orb_prompt = "Earned orbs: "
treats_prompt = "Jelly bear time!"
