{
  description = "devshell for unnamed-nn";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    forAllSystems = nixpkgs.lib.genAttrs [
      "aarch64-linux"
      "x86_64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];
  in {
    devShells = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default = pkgs.mkShell {
          packages = builtins.attrValues {
            inherit
              (pkgs)
              nixd
              alejandra
              cmake
              clang-tools
              ;
          };

          shellHook = ''
            export SHELL='${pkgs.mksh}/bin/mksh'
          '';
        };
      }
    );
  };
}
