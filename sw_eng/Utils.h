//#include <bits/stdc++.h> 
#include <string>
  

// This function generates the key in 
// a cyclic manner until it's length isi'nt 
// equal to the length of original text 
std::string generateKey(std::string str, std::string key);

// This function returns the encrypted text 
// generated with the help of the key 
std::string cipherText(std::string str, std::string key);

// This function decrypts the encrypted text 
// and returns the original text 
std::string originalText(std::string cipher_text, std::string key);
