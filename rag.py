import os


def main():
#    read_folder()
    chunking()

def read_folder():                                                                      #reading folders to give as output, then would be used for chunking.
    
    for folder , sub_folder , files in os.walk(r"D:\Workstation_rag_files"):            #walking into the folder and telling us which files are there.      
        if ".git" in sub_folder:
            sub_folder.remove(".git")                                                   
        for name in files:
            if name.endswith(".md"):
                full_path = os.path.join(folder,name)
                with open(full_path,encoding="utf-8") as file:                          #opnes the .md files.
                    text = file.read()


def chunking():
    text = ("This is a test for evaluating the chunking of the programm we wrote. This is a fixed size chunking programm.")
    chunks = text.split(" ")
    chunking_range = len(chunks)//3
    remaining = len(chunks)%3
    
    if remaining in (1,2):
        chunking_range += 1
        
    a , b = 0 , 3
    for _ in range(chunking_range):
        print(chunks[a:b])
        a += 3
        b += 3

if __name__ == "__main__":
    main()
