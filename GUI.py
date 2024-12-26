from client import *

connect_to_server()
root = Tk()
root.geometry("450x400")  # starting window size
root.configure(bg="#0078D4")  # GUI background color
create_form_elements(root)
root.mainloop()
