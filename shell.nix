{ pkgs ? import <nixpkgs> { } }:
with pkgs.python38Packages;
let
  mach-nix = import
    (
      builtins.fetchGit {
        url = "https://github.com/DavHau/mach-nix/";
        ref = "refs/tags/3.3.0";
      }
    )
    {
      inherit pkgs;
      pypiDataRev = "9699653ccac8681ce92004b210b73f5d929d6e41"; # 2021-08-30T20:17:04Z
      pypiDataSha256 = "0zayz2flj46gf6gc3h9ip9i2fkpff50qiyjnrrpcgxjyxnq4rndz";
    };
  custom-python = mach-nix.mkPython {
    python = "python38Full";
    requirements = ''
      gym[atari]
    '';
    _.atari-py = {
      nativeBuildInputs.add = with pkgs; [ cmake ];
      buildInputs.add = with pkgs; [zlib];
      dontUseCmakeConfigure = true;
    };
  };
in
with pkgs; mkShell {
  buildInputs = [
    custom-python
    pyglet
  ];
}
