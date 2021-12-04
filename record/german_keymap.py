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

code="de"


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
# no °, no dead keys, german layout
left_chars = ["1", "2", "3", "4", "5", "q", "w", "e", "r", "t", "a", "s", "d", "f", "g", "<", "y", "x", "c", "v", "b"]
left_upper_chars = ["!", '"', "§", "$", "%"] + [el.upper() for el in left_chars[5:15]] + [">"] + [el.upper() for el in left_chars[16:]]
right_chars = ["6", "7", "8", "9", "0", "ß", "z", "u", "i", "o", "p", "ü", "+", "h", "j", "k", "l", "ö", "ä", "#", "n", "m", ",", ".", "-"]
right_upper_chars = ["&", "/", "(", ")", "=", "?"] + [el.upper() for el in right_chars[6:12]] + ["*"] + [el.upper() for el in right_chars[13:19]] + ["'"] + [el.upper() for el in right_chars[20:22]] + [";", ":", "_"]

# pair left and right key characters with their shift-modified version
left_keys = tuple(zip(left_chars, left_upper_chars))
right_keys = tuple(zip(right_chars, right_upper_chars))
keys = tuple(left_keys + right_keys)

# common german layout (same keys as other layouts)
common_left_chars = left_chars.remove("<")
common_left_upper_chars = left_upper_chars.remove(">")
common_right_chars = right_chars.remove("#")
common_right_upper_chars = right_upper_chars.remove("'")
common_left_keys = tuple(zip(left_chars, left_upper_chars))
common_right_keys = tuple(zip(right_chars, right_upper_chars))
common_keys = tuple(left_keys + right_keys)

# define key codes
left_codes = [10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 38, 39, 40, 41, 42, 50, 94, 52, 53, 54, 55, 56]
right_codes = [15, 16, 17, 18, 19, 20, 22, 29, 30, 31, 32, 33, 34, 35, 36, 43, 44, 45, 46, 47, 48, 51, 57, 58, 59, 60, 61, 62]
space_code = [65]
keycodes = left_codes + right_codes + space_code

# helper texts for tasks
# input
continue_ready_prompt = "(Sobald du bereit bist, drücke bitte einmal die EINGABETASTE (RETURN) um fortzufahren...)"
continue_prompt = "(Bitte drücke einmal die EINGABETASTE (RETURN) um fortzufahren...)"
start_prompt = "(Bitte drücke einmal die EINGABETASTE (RETURN) um zu starten...)"
continue_or_stay_prompt = "(Bitte drücke die EINGABETASTE (RETURN) noch einmal um fortzufahren, oder eine beliebige Taste, um bei diesem Aufgabenteil zu bleiben...)"
repeat_prompt = "(Falls etwas schief gelaufen ist und du die letzte Aufgabe wiederholen möchtest, drücke \"r\"...)"

# initial
intro_prompt = """Einführung\n
Im Folgenden werden dir nacheinander einige Aufgaben präsentiert. Bitte lese dir die Anweisungen genau durch und befolge sie.
Aufgabentypen wiederholen sich. Du kannst dich an dem Titel orientieren, um die Texte zu überspringen, wenn du dich an die Anweisungen erinnerst.
Zwischen den Aufgaben gibt es keine Aufnahmen und die Anweisungen werden dir vorschlagen eine Pause einzulegen, wenn du nicht sofort fortfahren willst.
Solltest du von einer Aufgabe überfordert sein, eine Pause brauchen oder aufhören wollen, zögere nicht, es zu sagen.
Sollte etwas schiefgehen, können wir jederzeit von diesem Punkt wieder anfangen.\n
""" + start_prompt

# sync
sync_prompt = """Bitte klatsche einmal in die Hände.\n
Sobald ein Hinweis zum Fortfahren erscheint, wurde das Klatschen registriert und du kannst weitermachen.
Falls kein Hinweis erscheint, versuche es bitte noch einmal (die Bewegung muss prägnant aber darf nicht schmerzhaft sein, die Lautheit ist nicht relevant).
Wir müssen dies aufnehmen, da die Bewegung einen plötzlichen Halt beider Hände verursacht, über den wir die beiden Myos synchronisieren können.
Vielen Dank.\n
""" + continue_prompt

# before task
start_uniform_random_memory_task_prompt = """Verschwindende Zufallszeichen\n
Im Folgenden werden dir einige Gruppen zu kopierender Zeichen präsentiert, die verschwinden, sobald du eine Taste drückst.
Bitte präge dir gut ein, welche Zeichen du siehst, ehe du anfängst zu tippen.
Bitte beachte, dass für diese Aufgabe die Eingabetaste durch {} und die Leertaste durch {} angezeigt werden.
Die letzte Gruppe kann weniger Zeichen enthalten als die vorangegangenen Gruppen.
Deine Eingabe wird automatisch verarbeitet.\n
{}""".format(chr(8629), chr(9251), start_prompt)
start_uniform_random_task_prompt = """Bleibende Zufallszeichen\n
Im Folgenden werden dir einige Gruppen zu kopierender Zeichen präsentiert.
Bitte beachte, dass für diese Aufgabe die Eingabetaste durch {} und die Leertaste durch {} angezeigt werden.
Die letzte Gruppe kann weniger Zeichen enthalten als die vorangegangenen Gruppen.
Deine Eingabe wird automatisch verarbeitet.\n
{}""".format(chr(8629), chr(9251), start_prompt)
start_copy_text_prompt = """Text\n
Im Folgenden werden dir einige zu kopierende Texte gezeigt.
Deine Eingabe wird verarbeitet, sobald du die EINGABETASTE (RETURN) zweimal hintereinander drückst.
Bitte verwende die Eingabetaste nur, wenn du alles abgetippt hast - der Text wird am Rand automatisch umgebrochen.\n
""" + start_prompt
start_copy_pangram_prompt = """Pangramme\n
Im Folgenden werden dir einige zu kopierende Pangramme gezeigt (meist unsinnige Sätze, die jeden Buchstaben mindestens einmal enthalten).
Deine Eingabe wird verarbeitet, sobald du die EINGABETASTE (RETURN) zweimal hintereinander drückst.\n
""" + start_prompt
start_game_prompt = """Spiel\n
Die letzte Aufgabe besteht aus einem Spiel, das im Folgenden erklärt wird.
Am Ende der Erklärung gibt es eine Momentaufnahme des Spiels.
Navigiere durch die Erklärung und starte das Spiel, indem du die EINGABETASTE (RETURN) drückst.
Du kannst selbst bestimmen wann du aufhören möchtest - es gibt kein definiertes Ende.
Viel Spaß!\n
""" + start_prompt

# during task
copy_prompt = "Bitte kopiere folgende Zeichen:"
copy_input_prompt = "Bitte kopiere folgenden Text und drücke die EINGABETASTE (RETURN):"

# between tasks
end_task_prompt = """Vielen Dank!
Du kannst jetzt eine Pause einlegen, bis du für die nächste Aufgabe bereit bist.\n
""" + continue_ready_prompt + "\n" + repeat_prompt
end_task_with_posture_change_prompt = """Bitte verändere deine Haltung (Sitzposition, Handhaltung, Tastaturposition und/oder Körperhaltung).\n
Vielen Dank!
Du kannst jetzt eine Pause einlegen, bis du für die nächste Aufgabe bereit bist.\n
""" + continue_ready_prompt + "\n" + repeat_prompt
end_task_with_extended_break_prompt = """Vielen Dank!
Wie wäre es mit einer kurzen Pause? Steh einfach mal auf, geh ein paar Schritte, streck dich.\n
""" + continue_ready_prompt + "\n" + repeat_prompt

# after tasks
other_task = "Die Aufnahme wurde gestoppt bei Aufgabe "
other_task_cont = "Um mit der nächsten Aufgabe fortzufahren, starte das Programm mit denselben Argumenten und -c "
last_task = "Das war die letzte Aufgabe.\n"
outro_prompt = "Vielen Dank, dass du dir die Zeit genommen hast, uns bei unserer Forschung zu unterstützen!" + thank_you_cat + "\n"


# pw task specific texts
intro_prompt = """Einführung\n
Im Folgenden werden dir nacheinander einige Aufgaben präsentiert, in denen du \"Passwörter\" lernst, indem du sie wiederholt tippst. Wir wollen diese Daten nutzen, um zu sehen, wie gut unsere Machine Learning Modelle - die darauf trainiert sind, Tasten in den von der Myo aufgenommenen Daten zu erkennen und zu identifizieren - Passwörter rekonstruieren können.
Bitte lese dir die Anweisungen genau durch und befolge sie.
Die Anweisungen einer Aufgabe wiederholen sich für jede Aufgabe. Falls du dich an sie erinnerst, musst du sie nicht noch einmal lesen.
Aufgabentypen wiederholen sich. Du kannst dich an dem Titel orientieren, um die Texte zu überspringen, wenn du dich an die Anweisungen erinnerst.
Zwischen den Aufgaben gibt es keine Aufnahmen und die Anweisungen werden dir vorschlagen eine Pause einzulegen, wenn du nicht sofort fortfahren willst.
Solltest du von einer Aufgabe überfordert sein, eine Pause brauchen oder aufhören wollen, zögere nicht, es zu sagen.
Sollte etwas schiefgehen, können wir jederzeit von diesem Punkt wieder anfangen.\n
""" + start_prompt
random_task = "Zufällige "
pwgen_task = "pwgen-Generierte "
xkcd_task = "xkcd-Nachempfundene "
insecure_task = "Unsichere "
start_pw_task_prompt = """Passwörter\n
Im Folgenden werden dir einige Gruppen mehrfach zu kopierender Zeichen präsentiert.
Bitte beachte, dass für diese Aufgabe die Leertaste durch {} angezeigt wird.
Bitte drücke Return um fortzufahren. Dir wird angezeigt, ob das Passwort korrekt ist.
Drücke noch einmal Return, um zur nächsten Eingabe fortzufahren.\n
{}""".format(chr(9251), start_prompt)
copy_pw_prompt = "Bitte tippe folgendes Passwort (hier fett gedruckt):"
additional_pw_prompt = "(datenstudie) nutzer% continue\n Passworteingabe erforderlich um fortzufahren..." + for_science
task_changes = "Nichts besonderes hier...\n\n" + continue_prompt
task_changes_coin = "Orb erhalten!\n\n" + continue_prompt
task_changes_extra_coin = "Du hast eine Truhe gefunden.\nStell dir vor du öffnest sie in einer langen Zwischensequenz...\nDrei Orbs erhalten!\n\n" + continue_prompt
orb_prompt = "Verdiente Orbs: "
treats_prompt = "Gummibärchenzeit!"
