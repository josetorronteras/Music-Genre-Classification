FROM continuumio/miniconda3:4.8.2

ADD envs /tmp

RUN conda env create -f tmp/base_enviroment.yml && \
    conda env update -f tmp/conda_forge_dependencies.yml

ENV PATH /opt/conda/tmp/music_clasification/bin:$PATH

# Configuring access to Jupyter
RUN mkdir /opt/notebooks
#RUN /opt/conda/bin/jupyter notebook --generate-config --allow-root

RUN /bin/bash -c "source activate music_clasification && \
                python -m ipykernel install --user --name=music_clasification"

# Run Jupyter notebook as Docker main process
CMD /opt/conda/envs/music_clasification/bin/jupyter notebook --allow-root --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser