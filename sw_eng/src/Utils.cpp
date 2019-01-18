#include "Utils.h"

std::string generateKey(std::string str, std::string key) 
{ 
    int x = str.size(); 

    for (int i = 0; ; i++) 
    { 
        if (x == i) 
            i = 0; 
        if (key.size() >= str.size()) 
            break; 
        key.push_back(key[i]); 
    } 
    return key; 
} 
  
std::string cipherText(std::string str, std::string key) 
{ 
    std::string cipher_text; 
  
    for (unsigned int i = 0; i < str.size(); i++) 
    { 
        // converting in range 0-92 
        int x = (str[i] + key[i]) % 91; 
  
        // convert into alphabets(ASCII) 
        x += '#'; 
  
        cipher_text.push_back(x); 
    } 
    return cipher_text; 
} 
  
std::string originalText(std::string cipher_text, std::string key) 
{ 
    std::string orig_text; 
  
    for (unsigned int i = 0 ; i < cipher_text.size(); i++) 
    { 
        // converting in range 0-92 
        int x = (cipher_text[i] - key[i] + 91 + 21) % 91; 
  
        // convert into alphabets(ASCII) 
        x += '#'; 
        orig_text.push_back(x); 
    } 
    return orig_text; 
} 

