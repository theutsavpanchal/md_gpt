import shutil
import os

def empty_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) 
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def write_doclist(doclist, directory="docs"):
   empty_directory(directory)
   i=0
   for doc in doclist:
      i+=1
      content = doc.page_content
      with open(f"{directory}/doc{i}.txt", "w+", encoding="utf-8") as file:
         file.write(content)
         file.close()


def write_response(resp):
    with open("response.txt", "w+") as f:
        f.write(resp)
    f.close()

