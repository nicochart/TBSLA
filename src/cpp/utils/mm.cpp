#include <tbsla/cpp/MatrixCOO.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace tbsla { namespace utils { namespace io {

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  do {
    pos = str.find(delim, prev);
    if (pos == std::string::npos)
      pos = str.length();
    std::string token = str.substr(prev, pos-prev);
    if (!token.empty())
      tokens.push_back(token);
    prev = pos + delim.length();
  } while (pos < str.length() && prev < str.length());
  return tokens;
}

struct MatrixFormatReadException : public std::exception {
   const char * what () const throw () {
      return "This Matrix format import is not implemented!";
   }
};

tbsla::cpp::MatrixCOO readMM(std::string fname) {
  std::ifstream is(fname);
  std::string line;
  std::string delim(" ");

  if (is.is_open()) {
    getline(is, line);
    std::vector<std::string> splits = split(line, delim);
    std::cout << line << "\n";
    if(splits[0].compare(std::string("%%MatrixMarket")) == 0
       && splits[1].compare(std::string("matrix")) == 0
       && splits[2].compare(std::string("coordinate")) == 0
       && splits[3].compare(std::string("real")) == 0 ) {
      if(splits[4].compare(std::string("general")) == 0) {
        while(line.rfind("%", 0) == 0) {
          getline(is, line);
        }
        std::cout << line << "\n";
        std::stringstream ss(line);
        int nc, nr, nv;
        ss >> nr;
        ss >> nc;
        ss >> nv;
        tbsla::cpp::MatrixCOO m (nr, nc, nv);
        int r, c;
        double v;
        for(int i = 0; i < nv && !is.eof(); i++) {
          is >> r;
          is >> c;
          is >> v;
          m.push_back(r - 1, c - 1, v);
        }
        return m;
      } else if(splits[4].compare(std::string("symmetric")) == 0) {
        while(line.rfind("%", 0) == 0) {
          getline(is, line);
        }
        std::cout << line << "\n";
        std::stringstream ss(line);
        int nc, nr, nv;
        ss >> nr;
        ss >> nc;
        ss >> nv;
        tbsla::cpp::MatrixCOO m (nr, nc, nv * 2);
        int r, c;
        double v;
        for(int i = 0; i < nv && !is.eof(); i++) {
          is >> r;
          is >> c;
          is >> v;
          m.push_back(r - 1, c - 1, v);
          if(r != c)
            m.push_back(c - 1, r - 1, v);
        }
        return m;
      }
    }
  }
  throw MatrixFormatReadException();
}

}}}

