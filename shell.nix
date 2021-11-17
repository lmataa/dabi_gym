{ pkgs ? (import <nixpkgs> {}).pkgs }:
with pkgs.python39Packages;

pkgs.mkShell{
  # venvDir = "./.venvDir";
  buildInputs = [
	# poetry env
	pkgs.python39
	pkgs.poetry

	# python libs - rather basic
	numpy
	networkx
	pkgs.libspatialindex
	dateutil
	pandas
	Rtree

	# openai gym deps
	gym
	pyglet
	
	# jupyter nb
	jupyter
	ipykernel
	ipywidgets
	jupyterlab

	# Copilot
	pkgs.vscode
	pkgs.gnome.gnome-keyring
	];
}

