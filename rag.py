import os


def main():
    read_folder()


def read_folder():                                                                      #reading folders to give as output, then would be used for chunking.
    
    for folder , sub_folder , files in os.walk(r"D:\Workstation_rag_files"):            #walking into the folder and telling us which files are there.      
        if ".git" in sub_folder:
            sub_folder.remove(".git")                                                   
        for name in files:
            if name.endswith(".md"):
                full_path = os.path.join(folder,name)
                with open(full_path,encoding="utf-8") as file:                          #opnes the .md files.
                    print(file.read())                                                  #Later we'll call this function.

        
if __name__ == "__main__":
    main()
